# CNN Folder: Custom Neural Network ATT&CK Model

This folder contains a custom neural network training path implemented with:

- OneVsRestClassifier
- MLPClassifier (neural network)
- TruncatedSVD for sparse-to-dense compression

## Training script

- `train_custom_neural_network.py`

## Train command

```powershell
.\.venv\Scripts\python.exe CNN\train_custom_neural_network.py \
  --input labeled_ja4_flows_malicious_only.csv \
  --output-prefix CNN\artifacts\mitre_multilabel \
  --model-tag custom_mlp \
  --min-label-frequency 5 \
  --threshold 0.35
```

## Outputs

Generated under `CNN/artifacts/` with run-scoped names:

- `*_bundle.pkl`
- `*_metrics.json`
- `*_label_catalog.json`
- `*_test_predictions.csv`
- `*_resolved_labels.csv`
- `Resultsmetrics.txt`
