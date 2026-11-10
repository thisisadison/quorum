# quorum

A scoring framework for evaluating the quality of LLM-generated sentiment labels on financial text.

Large language models can label data quickly and cheaply, but their labels are not guaranteed to be correct. This project uses LLMs to label financial news sentences and develops a framework that scores how reliable each label is. The framework is validated against expert human annotations from the Financial PhraseBank.


## Dataset

[Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank) (Malo et al., 2014): English sentences from financial news, labelled negative / neutral / positive by annotators with finance backgrounds.

- **Subset:** 50% annotator agreement, 4,846 sentences. Restricting to full agreement would discard 53% of the data and keep only the least ambiguous sentences.
- **Classes:** neutral 59.4%, positive 28.1%, negative 12.5%.
- **Preparation:** duplicate sentences are removed, each sentence gets a stable hashed ID, and the data is split 85/15 into a labelling pool and a test set, stratified by label (seed 42).

The data is downloaded from Hugging Face at runtime and is not stored in this repository. Alternative datasets that were considered and rejected are documented in `notebooks/dataset_selection.ipynb`.

## Repository structure

```
quorum/
├── configs/
│   └── prompts/
│       └── prompt_v1.txt        # LLM labelling prompt
├── notebooks/
│   ├── dataset_selection.ipynb  # candidate datasets and selection rationale
│   └── fyp_eda.ipynb            # exploratory data analysis of Financial PhraseBank
├── scripts/
│   └── run_prepare_data.py      # entry point for data preparation (in progress)
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

## References

Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014). Good debt or bad debt: Detecting semantic orientations in economic texts. *Journal of the Association for Information Science and Technology*, 65(4), 782–796.
