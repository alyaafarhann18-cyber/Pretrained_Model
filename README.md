# 📋Bahrain Restaurant Review Analyzer — Pretrained Model Challenge

Restaurants in Bahrain wants to automatically understand visitor
reviews written in a **natural bilingual mix of English and Bahraini Gulf-dialect Arabic**:
overall sentiment and the main topic discussed, for the same review. Success is measured
against a small, hand-labeled evaluation set.

Built as part of the **Pretrained Model Challenge lab** (General Assembly Data Science program, in partnership with BIBF).


## Data Collection
**60 restaurant reviews** in a natural bilingual mix (34 English, 26 Bahraini-dialect Arabic), labeled along **two independent dimensions**:
- `reviews_sentiment.csv` — 60 reviews, labeled positive/negative/neutral (Task A).
- `reviews_topics.csv` — 60 reviews, labeled with one of 7 topics (Task B).

## Data Preparation
Minimal cleaning only (whitespace stripping, dropping empty rows). A language flag (`en`/`ar`)
is derived per row using an Arabic-script regex check, used throughout evaluation to break down
results by language.

## Model Selection

- **Language-routed:** `cardiffnlp/twitter-roberta-base-sentiment-latest` (English) +
  `CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment` (Arabic dialect), chosen per-review by
  detected language.
- **Single multilingual model:** `nlptown/bert-base-multilingual-uncased-sentiment`, used for
  every review regardless of language.

For zero-shot topic classification, `joeddav/xlm-roberta-large-xnli` is used for
both languages with one shared set of English candidate labels.



## Launch the app:

https://pretrainedmodel-boo5jztsx6hyxtx9vjmvvc.streamlit.app/


