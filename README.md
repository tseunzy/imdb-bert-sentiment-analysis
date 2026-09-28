# IMDb Sentiment Classification with BERT

Fine-tuning **`bert-base-uncased`** for binary movie-review sentiment classification using **PyTorch** and **Hugging Face Transformers**.

The saved experiment achieves **88.66% accuracy** and **88.71% F1** on the 25,000-review IMDb test set.

## Project overview

[IMDB_BERT_SENTI.ipynb](IMDB_BERT_SENTI.ipynb) demonstrates the complete workflow: loading IMDb, tokenizing reviews, building a classification head on pretrained BERT, fine-tuning, selecting a checkpoint using validation accuracy, evaluating on the test set, and predicting sentiment for new text.

Sentiment labels are **0 = Negative** and **1 = Positive**.

## Dataset

The notebook loads `stanfordnlp/imdb` with Hugging Face Datasets. It splits the original training partition using `test_size=0.1` and `seed=42`, keeping the original test partition for final evaluation.

| Partition | Reviews |
|---|---:|
| Training | 22,500 |
| Validation | 2,500 |
| Test | 25,000 |

`BertTokenizer` converts reviews into WordPiece tokens, adds special tokens, and creates input IDs, attention masks, and token-type IDs. Sequences are padded or truncated to **128 tokens**, including special tokens. Attention masks identify padding positions.

## Model architecture

```text
Review text
    → BERT tokenizer
    → Pretrained bert-base-uncased
    → Final hidden state of the [CLS] token (768 features)
    → Dropout (0.3)
    → Linear layer (768 → 2)
    → Negative / positive logits
```

The model uses the final `[CLS]` hidden state rather than BERT's pooler output. BERT is not frozen; its encoder is fine-tuned alongside the classification head. The notebook reports **109,483,778 parameters** with `requires_grad=True`.

Training uses raw logits with `CrossEntropyLoss`. For custom-text inference, softmax converts the logits into class scores, and the highest-scoring class determines the prediction.

## Training configuration

| Setting | Value |
|---|---|
| Pretrained model | `bert-base-uncased` |
| Maximum sequence length | 128 |
| Batch size | 8 |
| Epochs | 3 |
| Optimizer | AdamW |
| Learning rate | `2e-5` |
| Loss | Cross-entropy |
| Classifier dropout | 0.3 |
| Checkpoint selection | Highest validation accuracy |
| Checkpoint filename | `bert_imdb_best.pt` |
| Device | CUDA when available; otherwise CPU |

The notebook runs all three epochs and saves the model whenever validation accuracy improves. It then restores the selected checkpoint before test evaluation.

## Results

### Training history

| Epoch | Training loss | Training accuracy | Validation loss | Validation accuracy |
|---|---:|---:|---:|---:|
| 1 | 0.3354 | 85.08% | 0.2768 | **88.88%** |
| 2 | 0.1832 | 92.81% | 0.3035 | 87.84% |
| 3 | 0.0854 | 96.99% | 0.3902 | 88.16% |

**Epoch 1** produced the best validation accuracy. Later epochs increased training accuracy while validation loss rose, indicating overfitting.

### Test performance

The restored epoch-1 checkpoint produced the following saved results:

| Metric | Value |
|---|---:|
| Accuracy | 88.66% |
| Precision | 88.36% |
| Recall | 89.06% |
| F1 score | 88.71% |

Precision, recall, and F1 use positive sentiment as the positive class.

| Actual / Predicted | Negative | Positive |
|---|---:|---:|
| Negative | 11,033 | 1,467 |
| Positive | 1,367 | 11,133 |

The notebook also plots the confusion matrix and training/validation curves. These figures and metrics describe the saved run; training again can produce different results.

## Setup and execution

The recorded notebook environment uses **Python 3.10.12**. Required packages are `torch`, `transformers`, `datasets`, `tqdm`, `scikit-learn`, and `matplotlib`, plus Jupyter for running the notebook.

Create and activate an environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate` instead. `ipywidgets` supports notebook progress bars.

From the directory containing the notebook, launch:

```bash
jupyter lab IMDB_BERT_SENTI.ipynb
```

Select the environment's Python kernel and run the cells in order. Initial execution downloads the IMDb dataset, tokenizer, and pretrained BERT weights unless already cached. The code supports CPU execution; a CUDA-capable GPU can accelerate fine-tuning, provided the installed PyTorch build supports the local GPU environment.

The requirements file is a dependency starting point, not a verified lockfile. Exact package versions are not pinned in this project.

## Predict sentiment

After loading the selected checkpoint and defining the notebook's `predict_sentiment()` function:

```python
text = "This movie was absolutely fantastic."
sentiment, negative_prob, positive_prob = predict_sentiment(text)

print("Prediction:", sentiment)
print(f"Negative probability: {negative_prob:.4f}")
print(f"Positive probability: {positive_prob:.4f}")
```

Saved example predictions include:

| Review | Prediction | Positive score |
|---|---|---:|
| This movie was absolutely fantastic. | Positive | 0.9981 |
| This movie was not good. | Negative | 0.0091 |
| This movie was terrible. | Negative | 0.0058 |
| This movie was not terrible, i don't hate it | Positive | 0.9314 |

These examples illustrate the model's behavior; they do not establish reliable handling of every form of negation or mixed sentiment.

## Saving and loading the model

The trained checkpoint is excluded from this repository. Run the training cells to generate `bert_imdb_best.pt` before loading it or running predictions.

The training loop saves the best model's parameter state:

```python
torch.save(model.state_dict(), "bert_imdb_best.pt")
```

To use an existing checkpoint without retraining, run the import/device setup, tokenizer initialization, and `BertSentimentClassifier` definition, then restore the weights:

```python
model = BertSentimentClassifier().to(device)
model.load_state_dict(
    torch.load("bert_imdb_best.pt", map_location=device, weights_only=True)
)
model.eval()
```

Define `predict_sentiment()` afterward to classify new text. Use the same `bert-base-uncased` tokenizer and maximum length of 128.

The checkpoint path is relative to the notebook kernel's working directory. This file stores model weights; it does not include the tokenizer, class definition, optimizer state, or training history. Reconstructing the classifier also loads the base BERT model from its cache or downloads it.

## Limitations and future work

- **Truncated reviews:** text beyond 128 tokens is discarded, potentially removing sentiment-bearing content.
- **Overfitting:** validation performance peaks in the first epoch. Further experiments could investigate regularization and training schedules using the validation set.
- **Reproducibility:** the data split uses seed 42, but model initialization and training randomness are not fully seeded. Record dependency versions and repeat runs to measure variation.
- **Loss reporting:** epoch losses average batch means rather than weighting by batch size, so the smaller final training and validation batches receive extra weight. Accuracy and final test classification metrics are computed per example.
- **Domain coverage:** evaluation covers IMDb reviews; performance on other writing styles or domains has not been measured.
- **Probability calibration:** softmax scores are not demonstrated to be calibrated confidence estimates.

Possible extensions include longer input sequences, systematic error analysis, and comparisons with recurrent models or a Transformer trained from scratch under matching experimental conditions. Keep the test set reserved for final evaluation when selecting changes.
