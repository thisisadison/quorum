"""LLM labelling: record raw model outputs, no signal calculations.

For each sentence, one pass through the model gives us:
  - the probability of each label (negative / neutral / positive)
  - the log-probability the model gave each word of the sentence
    (signals such as perplexity are calculated from these later, at scoring)
"""

import gc
import json
import os
import time

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LABELS = ["negative", "neutral", "positive"]

LOW_MASS = 0.90        # a sentence is flagged if the 3 labels get less than 90% of the probability
MAX_LOW_SHARE = 0.01   # the run stops if more than 1% of sentences are flagged

# every way a model might write each label as the first word of its answer
LABEL_FORMS = {
    "negative": ["negative", "Negative", " negative", " Negative"],
    "neutral":  ["neutral",  "Neutral",  " neutral",  " Neutral"],
    "positive": ["positive", "Positive", " positive", " Positive"],
}


# ---------------------------------------------------------------------------
# loading and unloading models
# ---------------------------------------------------------------------------

def load_model(model_name):
    """download (first time only) and load a model onto the GPUs"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map="auto")
    return model, tokenizer


def free_gpu():
    """clear GPU memory so the next model fits, then print what is left"""
    gc.collect()
    torch.cuda.empty_cache()
    for i in range(torch.cuda.device_count()):
        used = torch.cuda.memory_allocated(i) / 1e9
        print(f"GPU {i}: {used:.1f} GB in use")


# ---------------------------------------------------------------------------
# label tokens
# ---------------------------------------------------------------------------

def first_token_id(tokenizer, word):
    """id of the first token of a word. some tokenizers turn a leading space
    into its own token; in that case take the token after it"""
    tokens = tokenizer.encode(word, add_special_tokens=False)
    first_piece = tokenizer.decode([tokens[0]])
    if first_piece.strip() == "" and len(tokens) > 1:
        return tokens[1]
    return tokens[0]


def get_label_token_ids(tokenizer):
    """for each label, the set of token ids of all its forms
    e.g. {"positive": {id of 'positive', id of 'Positive', ...}, ...}"""
    label_ids = {}
    for label in LABELS:
        ids = set()                      # a set, so the same id is never counted twice
        for form in LABEL_FORMS[label]:
            ids.add(first_token_id(tokenizer, form))
        label_ids[label] = ids
    return label_ids


def check_label_tokens(tokenizer):
    """CHECK: stop if a token would be counted for two different labels"""
    label_ids = get_label_token_ids(tokenizer)

    all_ids = []
    for label in LABELS:
        all_ids.extend(label_ids[label])

    if len(all_ids) != len(set(all_ids)):
        raise ValueError(f"two labels share a token: {label_ids}")


# ---------------------------------------------------------------------------
# one sentence: label probabilities and token log-probabilities
# ---------------------------------------------------------------------------

def build_model_input(sentence, prompt_template, tokenizer):
    """put the sentence into the prompt, wrap it in the model's chat format,
    and turn it into tokens"""
    prompt = prompt_template.format(text=sentence)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,   # the model's answer comes next
        enable_thinking=False,        # no reasoning text before the answer (qwen)
    )

    # the chat template already adds the start token, so the tokenizer must not add a second one.
    # return_offsets_mapping gives each token's character position in the text (used to find the sentence tokens)
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded.pop("offset_mapping")[0].tolist()
    return text, encoded, offsets


def label_probabilities(next_token_scores, label_ids):
    """probability of each label for the first word of the answer,
    adding up all forms of each label"""
    probs = torch.softmax(next_token_scores.float(), dim=-1)

    label_probs = {}
    for label in LABELS:
        total = 0.0
        for token_id in label_ids[label]:
            total += probs[token_id].item()
        label_probs[label] = total
    return label_probs


def sentence_token_logprobs(all_scores, token_ids, offsets, text, sentence):
    """the log-probability the model gave each actual token of the sentence.

    the model predicts every token from the tokens before it, so the scores at
    position i-1 are its prediction for token i. we look up how likely it
    thought each real sentence token was. values are <= 0: close to 0 means
    expected, very negative means surprising."""
    sentence_start = text.rfind(sentence)
    if sentence_start < 0:
        raise ValueError("sentence not found in the prompt")
    sentence_end = sentence_start + len(sentence)

    token_logprobs = []
    for i, (char_start, char_end) in enumerate(offsets):
        is_in_sentence = char_end > sentence_start and char_start < sentence_end and char_end > char_start
        if i == 0 or not is_in_sentence:
            continue
        log_probs = torch.log_softmax(all_scores[i - 1].float(), dim=-1)   # prediction made just before token i
        token_logprobs.append(log_probs[token_ids[i]].item())             # how likely the actual token was
    return token_logprobs


def get_label_probs(sentence, prompt_template, model, tokenizer, label_ids):
    """one forward pass for one sentence. returns (label probabilities, sentence token log-probabilities)"""
    text, encoded, offsets = build_model_input(sentence, prompt_template, tokenizer)
    model_inputs = {name: tensor.to(model.device) for name, tensor in encoded.items()}

    with torch.no_grad():                                   # not training, so no gradients needed
        all_scores = model(**model_inputs).logits[0]        # one row of scores per input token

    next_token_scores = all_scores[-1]                      # scores for the first word of the answer
    label_probs = label_probabilities(next_token_scores, label_ids)

    token_ids = model_inputs["input_ids"][0]
    token_logprobs = sentence_token_logprobs(all_scores, token_ids, offsets, text, sentence)
    return label_probs, token_logprobs


def normalise(probs):
    """rescale the 3 probabilities to add up to 1.
    also return how much they added up to before rescaling (the label mass)"""
    mass = sum(probs.values())

    scaled = {}
    for label in probs:
        scaled[label] = probs[label] / mass

    return scaled, mass


# ---------------------------------------------------------------------------
# a whole dataset
# ---------------------------------------------------------------------------

def load_checkpoint(out_path):
    """rows already saved by an earlier, interrupted run (empty if none)"""
    if out_path is None or not os.path.exists(out_path):
        return []
    rows = pd.read_csv(out_path).to_dict("records")
    print(f"resuming: {len(rows)} sentences already labelled")
    return rows


def label_dataset(df, prompt_template, model, tokenizer, out_path=None, save_every=250):
    """label every sentence in df, one row of results per sentence.
    with out_path: saves progress every save_every sentences, and skips
    sentences already labelled if the run is restarted"""
    label_ids = get_label_token_ids(tokenizer)
    rows = load_checkpoint(out_path)
    done = set()
    for row in rows:
        done.add(row["item_id"])

    start = time.time()
    for n, (_, item) in enumerate(df.iterrows(), start=1):
        if item["item_id"] in done:
            continue

        probs, token_logprobs = get_label_probs(item["sentence"], prompt_template, model, tokenizer, label_ids)
        probs, mass = normalise(probs)
        best_label = max(probs, key=probs.get)

        rows.append({
            "item_id": item["item_id"],
            "negative": probs["negative"],
            "neutral": probs["neutral"],
            "positive": probs["positive"],
            "llm_label": best_label,
            "label_mass": mass,
            "low_mass": mass < LOW_MASS,
            "token_logprobs": json.dumps(token_logprobs),   # a list, stored as text in the CSV
        })

        if out_path is not None and len(rows) % save_every == 0:
            pd.DataFrame(rows).to_csv(out_path, index=False)          # checkpoint
        if n % 500 == 0:
            print(f"  {n}/{len(df)} sentences, {time.time() - start:.0f}s")

    results = pd.DataFrame(rows)
    if out_path is not None:
        results.to_csv(out_path, index=False)
    return results


def check_label_mass(results):
    """CHECK: stop if any number is missing, or if too many sentences put
    too little probability on the 3 labels"""
    share_low = results["low_mass"].mean()
    print(f"Lowest label_mass: {results['label_mass'].min():.4f}, Flagged: {share_low:.1%}")

    if results["label_mass"].isna().any():
        raise ValueError("some probabilities are NaN (float16 problem)")
    for logprobs in results["token_logprobs"]:
        values = json.loads(logprobs)
        if len(values) == 0 or any(v != v for v in values):     # v != v is only true for NaN
            raise ValueError("some sentences have missing or NaN token log-probabilities")
    if share_low > MAX_LOW_SHARE:
        raise ValueError(f"{share_low:.1%} of sentences have low label mass")
