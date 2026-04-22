# NLP Medical Condition Classifier

Classifies drug reviews by medical condition using three approaches: TF-IDF + SVC, Bi-LSTM, and BioBERT.

## Dataset

[Drug Review Dataset](https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018) — 161k patient reviews from Drugs.com.

Filtered to top 10 conditions (Anxiety, Birth Control, Pain, Depression, Acne, Bipolar Disorder, Insomnia, Weight Loss, Obesity, ADHD), balanced via stratified sampling. Final: ~29k train / ~22k test.

## Models

| Model | Test Accuracy | Weighted F1 |
|---|---|---|
| TF-IDF + SVC | 82.21% | 0.87 |
| Bi-LSTM (FastText 300d) | 83.83% | 0.84 |
| BioBERT v1.1 | **87.78%** | **0.86** |

## Notebooks

| File | Description |
|---|---|
| `0-explore-dataset.ipynb` | EDA and preprocessing |
| `1-svc-classification.ipynb` | TF-IDF vectorization + LinearSVC with GridSearchCV |
| `2-bi-lstm-classification.ipynb` | Bidirectional LSTM with pretrained FastText embeddings |
| `3-biobert-classification.ipynb` | Fine-tuned BioBERT (PubMed/PMC), run on Colab T4 |
| `4-errors-analysis.ipynb` | Misclassification analysis |

## Notes

- SVC uses stop word removal + lemmatization; BERT needs minimal preprocessing
- Hardest to distinguish: *Obesity* vs *Weight Loss* (semantically similar reviews)
- SVC is 30× faster to train with only ~5pp accuracy gap vs BioBERT
