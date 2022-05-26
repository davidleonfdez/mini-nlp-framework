# mini-nlp-framework
Simple NLP framework for creating classification and language modeling baselines

Other features include:
- Cross validation
- Dynamic training length criteria

# Installation

`pip install git+https://github.com/davidleonfdez/mini-nlp-framework.git`

# Examples

## Classification

This example assumes your data is stored in a csv file that has, at least, the following columns:
- "text": column that contains the documents/sequences
- "label": column of numeric type that contains the labels

For instance:

| index | text         | label |
| ----- | ------------ | ----- |
| 0     | I am happy   | 1     |
| 1     | I am sad     | 0     |
| 2     | I feel great | 1     |

A small classifier composed by a pretrained embedding and a linear layer is evaluated using k-fold cross-validation 
with k = 3 (because of the default parameter `n_folds = 3`).
For every fold, the model is trained until 3 epochs pass without an improvement of the previous best validation metric.

```
from mininlp.data import encode_text
from mininlp.metrics import BinaryClassificationMetric, MulticlassClassificationMetric
from mininlp.models import QuickClassifierProvider
from mininlp.train import MetricsPrinter, TrainLengthBestMetricEpochsAgo
from mininlp.validate import cross_validate

model_provider_cls = QuickClassifierProvider
df = pd.read_csv('/path/to/your/data.csv')
text_col = df.text
label_col = df.target
encoded_inputs = encode_text(text_col, model_provider_cls.embedding_source)
y = label_col.to_numpy()
n_classes = label_col.nunique()
max_seq_len = max(encoded_inputs.seq_lengths)
model_provider = model_provider_cls(encoded_inputs.vocab, max_seq_len, n_classes)
metric = BinaryClassificationMetric() if n_classes < 3 else MulticlassClassificationMetric()

cv_results = cross_validate(
    model_provider, encoded_inputs.X_padded, y, metric=metric, callbacks=[MetricsPrinter()],
    train_length=TrainLengthBestMetricEpochsAgo(3, lower_is_better=metric.lower_is_better)
)
print(cv_results)
```

## Language modeling

This example assumes your data is stored in a csv file that has, at least, a "text" column that contains the 
documents/sequences.

A language model composed by a transformer encoder and a linear layer is evaluated using k-fold cross-validation 
with k = 3 (because of the default parameter `n_folds = 3`).
For every fold, the model is trained for 10 epochs.

```
from mininlp.data import encode_text
from mininlp.metrics import LanguageModelMetric
from mininlp.models import CustomLanguageModelProvider
from mininlp.train import MetricsPrinter
from mininlp.validate import cross_validate

model_provider_cls = CustomLanguageModelProvider
df = pd.read_csv('/path/to/your/data.csv')
text_col = df.text
encoded_inputs = encode_text(text_col, model_provider_cls.embedding_source, min_seq_len=2)
X = encoded_inputs.X_padded[:, :-1]
y = encoded_inputs.X_padded[:, 1:]
max_seq_len = max(encoded_inputs.seq_lengths)

model_provider = model_provider_cls(encoded_inputs.vocab, max_seq_len)
n_epochs = 10
cv_results = cross_validate(
    model_provider, X, y, encoded_inputs.seq_lengths, metric=LanguageModelMetric(pad_idx=encoded_inputs.vocab.pad_idx),
    callbacks=[MetricsPrinter()], train_length=n_epochs
)
print(cv_results)
```

