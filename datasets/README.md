# Datasets

This project keeps only lightweight sample data and metadata files in GitHub.
Large raw and processed training datasets stay on disk locally but are ignored by Git so pushes remain reliable.

## Tracked In Git

- `datasets/README.md`
- `datasets/external/twitter_sentiment.csv`
- `datasets/processed/final_sample.csv`
- `datasets/processed/youtube_sample.csv`
- `datasets/raw/archive/*_category_id.json`

## Kept Local Only

The following files are intentionally ignored because they make the repository too large for normal GitHub pushes:

- `datasets/raw/archive/*videos.csv`
- `datasets/processed/final_dataset.csv`
- `datasets/processed/merged_sentiment_data.csv`
- `datasets/processed/youtube_cleaned.csv`
- `datasets/processed/sentiment_training_data.csv`

## If You Need The Full Data

Store the full datasets outside GitHub and regenerate them locally when needed. Good options are:

- Kaggle or the original public dataset source
- Google Drive or OneDrive shared link
- Git LFS, if you explicitly want large data versioned

## Recommended Workflow

1. Keep code, configs, model definitions, and small samples in Git.
2. Keep raw datasets and heavy processed outputs out of Git.
3. Document where full data comes from so others can reproduce the project locally.
