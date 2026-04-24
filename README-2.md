# Earnings Call Sentiment Analysis: FinBERT vs Traditional NLP

This repository contains an NLP project on **earnings-call sentiment analysis** with a direct comparison between **traditional bag-of-words / dictionary-based methods** and **FinBERT**, a finance-domain transformer model.

The project studies which sentiment measures best explain the **abnormal stock-market reaction after an earnings call**.

## Project Summary

The workflow compares multiple sentiment extraction approaches on earnings-call transcripts:

- **Harvard General Inquirer** dictionary
- **Loughran-McDonald (LM)** finance dictionary
- **Negation-adjusted** dictionary scoring
- **Custom finance-aware dictionary** built from the corpus
- **FinBERT** benchmark using `ProsusAI/finbert`

Instead of forcing a single text source, the project evaluates three transcript sections:

- **Presentation only**
- **Management answers in Q&A**
- **Full call transcript**

The final conclusion is that **full-call text is consistently the most informative corpus**, and that **FinBERT is highly competitive**, but a carefully tuned **custom finance-aware bag-of-words approach remains extremely strong and more interpretable**.

## Key Findings

Based on the report and notebook outputs:

- The **full-call corpus** performs better than presentation-only or Q&A-only variants.
- **Finance-specific sentiment methods** outperform general-purpose sentiment dictionaries.
- The **custom refined full-call dictionary** is the strongest bag-of-words approach across most return specifications.
- **FinBERT full-call sentiment** is the best transformer-based specification and slightly outperforms the top bag-of-words model for **CAR01-ff3**, while remaining slightly weaker on the other main return outcomes.

## Repository Contents

```text
.
├── earnings_call_sentiment_analysis.ipynb   # Main notebook
├── project_report.pdf                       # Project write-up
├── README.md                               # Repository overview
├── requirements.txt                        # Python dependencies
├── .gitignore                              # Ignore rules for notebooks, caches, models
└── LICENSE                                 # MIT license
```

## Methodology Overview

### 1. Data Construction
The notebook merges three course-provided datasets:

- call-level event data
- presentation transcript data
- segmented Q&A transcript data

Using `file_name` as the merge key, it builds one row per call and creates these text corpora:

- `presentation_text`
- `qa_answers_text`
- `full_call_text`

### 2. Text Preprocessing
For dictionary-based scoring, the notebook:

- lowercases text
- removes line breaks and punctuation
- normalizes whitespace
- counts words for length normalization

Tone is computed as:

```text
(positive_count - negative_count) / total_words
```

### 3. Traditional NLP Approaches
The notebook implements:

- Harvard sentiment dictionary
- Loughran-McDonald finance dictionary
- negation-aware polarity flipping
- a custom dictionary derived from frequent corpus words and finance-relevant stems

### 4. FinBERT Benchmark
The transformer benchmark uses `ProsusAI/finbert` from Hugging Face.

Because earnings-call transcripts are long, the notebook:

- splits transcripts into chunks of 220 words
- scores each chunk using FinBERT
- aggregates chunk-level probabilities with length-based weighting
- defines transcript sentiment as:

```text
FinBERT Tone = P(positive) - P(negative)
```

### 5. Regression Analysis
The project evaluates whether sentiment predicts post-call abnormal returns using OLS with HC3 robust standard errors.

Main outcomes:

- `CAR01-ff3`
- `CAR01-Carhart`
- `CAR-11-ff3`
- `CAR-11-Carhart`

Controls include:

- earnings surprise (`surp`)
- historical volatility (`hvol`)
- implied volatility (`IV_l1d`)
- firm size (`log_atq`)

## Main Results

### Best Bag-of-Words Result
The strongest bag-of-words specification is:

- **Custom refined dictionary on the full-call corpus**

### Best FinBERT Result
The strongest FinBERT specification is:

- **FinBERT on the full-call corpus**

### Overall Comparison
- FinBERT is a strong contextual benchmark.
- The custom finance-aware dictionary remains slightly stronger across most return specifications.
- The project shows that **simple methods can still be highly effective when the lexicon and corpus design are well chosen**.

## Tech Stack

- Python
- pandas
- numpy
- statsmodels
- PyTorch
- transformers
- sentencepiece
- accelerate
- Jupyter Notebook / Google Colab

## How to Run

### Option 1: Run locally
```bash
pip install -r requirements.txt
jupyter notebook
```
Then open:

```text
earnings_call_sentiment_analysis.ipynb
```

### Option 2: Run in Google Colab
Upload the notebook to Colab and run the cells. The notebook already includes package installation for the transformer section.

## Notes

- The notebook currently loads the course datasets and dictionaries from remote URLs.
- If those URLs change or become unavailable, you will need to replace them with local files or updated links.
- Depending on your machine, FinBERT scoring may take time, especially on CPU.

## Suggested Future Improvements

- split the workflow into reusable Python scripts
- add plots for model comparison and coefficient summaries
- cache FinBERT outputs to avoid repeated inference
- move data-loading URLs into a config file
- add a `src/` folder for cleaner project organization

## License

This repository is released under the MIT License.
