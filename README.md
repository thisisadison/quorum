# quorum

A scoring framework for evaluating the quality of LLM-generated sentiment labels on financial text.

Large language models can label data quickly and cheaply, but their labels are not guaranteed to be correct. This project uses LLMs to label financial news sentences and develops a framework that scores how reliable each label is. The framework is validated against expert human annotations from the Financial PhraseBank.


## Dataset

[Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank) (Malo et al., 2014): English sentences from financial news, labelled negative / neutral / positive by annotators with finance backgrounds.

- **Subset:** 50% annotator agreement, 4,846 sentences. Restricting to full agreement would discard 53% of the data and keep only the least ambiguous sentences.
- **Classes:** neutral 59.4%, positive 28.1%, negative 12.5%.
- **Preparation:** duplicate sentences are removed, each sentence gets a stable hashed ID, and the data is split 85/15 into a labelling pool and a test set, stratified by label (seed 42).

The data is downloaded from Hugging Face at runtime and is not stored in this repository. Alternative datasets that were considered and rejected are documented in `notebooks/dataset_selection.ipynb`.

## LLM labelling

Each pool sentence is labelled by three open-weight LLMs from different developers, so that their agreement is not driven by shared training:

| Model | Developer |
|---|---|
| [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | Alibaba |
| [Gemma-2-9B-it](https://huggingface.co/google/gemma-2-9b-it) | Google |
| [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | Meta |

- **Prompt:** every model receives the same prompt (`configs/prompts/prompt_v1.txt`), which asks for a single word: negative, neutral or positive.
- **Probabilities, not text:** the model reads the prompt once, and the probability of each label is taken from the first token of its answer. All written forms of a label (e.g. `positive`, `Positive`) are summed, since models differ in which form they prefer. The three probabilities are rescaled to sum to 1, and the highest is the model's label.
- **Checks:** a run stops if two labels share a token, if any probability is NaN, or if more than 1% of sentences put less than 90% of the probability on the three labels.
- **Pilot:** the method was tested on 100 pool sentences before the full run (`notebooks/llm_pilot.ipynb`). Mistral-7B-Instruct-v0.3 was tested and replaced, as its accuracy was below always predicting neutral and its confidence was the same whether right or wrong. Granite-3.3-8B-Instruct was dropped, as its answers often start with a tag, so the label cannot be read from the first token.

## Repository structure

```
quorum/
├── configs/
│   └── prompts/
│       └── prompt_v1.txt        # LLM labelling prompt
├── data/                        # created locally, not in the repository
│   ├── processed/               # pool.csv and test.csv
│   └── pilot/                   # LLM pilot results, one CSV per model
├── notebooks/
│   ├── dataset_selection.ipynb  # candidate datasets and selection rationale
│   ├── fyp_eda.ipynb            # exploratory data analysis of Financial PhraseBank
│   ├── llm_pilot.ipynb          # LLM labelling pilot (runs on Kaggle GPU)
│   └── agreement_analysis.ipynb # agreement between the LLMs on the pilot
├── scripts/
│   └── run_prepare_data.py      # creates the pool/test split
├── src/
│   └── quorum/
│       └── data_prep.py         # loading, deduplication and pool/test split
└── requirements.txt
```

## Setup

Requires Python 3.12.

1. Clone the repository
   ```bash
   git clone https://github.com/thisisadison/quorum
   cd quorum
   ```

2. Create and activate a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

NLTK stopwords are downloaded automatically when the EDA notebook runs.

## Usage

**Notebooks:** open `notebooks/dataset_selection.ipynb` or `notebooks/fyp_eda.ipynb` in Jupyter or VS Code, select the `venv` kernel, and run all cells.

**Data preparation:** imports takala financial phrase bank, create custom hash id, remove duplicates, perform pool/test split, convert pandas df into csv.
```bash
PYTHONPATH=src python scripts/run_prepare_data.py
```

**LLM labelling pilot:** `notebooks/llm_pilot.ipynb` needs a GPU and runs on Kaggle (GPU T4 x2).
1. Upload `data/processed/pool.csv` as a private Kaggle dataset and add it to the notebook.
2. Accept the model terms for Gemma and Llama on Hugging Face, create a read token, and add it in Kaggle under Add-ons → Secrets as `HF_TOKEN`.
3. Set the accelerator to GPU T4 x2 and run all cells. Results are saved to `/kaggle/working/`; download them into `data/pilot/`.

**Agreement analysis:** open `notebooks/agreement_analysis.ipynb` locally once the pilot CSVs are in `data/pilot/`.

## References

Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014). Good debt or bad debt: Detecting semantic orientations in economic texts. *Journal of the Association for Information Science and Technology*, 65(4), 782–796.
