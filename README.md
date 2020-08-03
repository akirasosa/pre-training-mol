# Applying self pre-training method to GNN for quantum chemistry

This repository contains some experiments for [*Applying self pre-training method to GNN for quantum chemistry*](https://medium.com/vitalify-asia/applying-self-pre-training-method-to-gnn-for-quantum-chemistry-7933e4e40a6?sk=0a2a9b114cbeb524679a54cc3ab63527).

## Prerequisites

* Python 3.8

```
# Getting data
curl -sSL "https://www.dropbox.com/s/fifvs2gpdnocxxr/qm9.parquet?dl=1" > ./data/qm9.parquet
```

## Self pre-training

Move under src/.

Estimate epochs to train.
```
python run_pre_train.py params/pre_train/001.yaml
```

Train with all data. It will take 200 epochs.
```
python run_pre_train.py params/pre_train/002.yaml
```

## Main training

Move under src/.

Train without self pre-training (baseline).
```
python run_train.py params/train/001.yaml
```

Edit config to use pre-trained weight.
```yaml
# src/params/train/002.yaml
pretrained_ckpt_path: xxx
```

Train with self pre-training.
```
python run_train.py params/train/002.yaml
```

## Results

|Pre-train|Target|Unit|MAE|Epochs|
|---|---|---|---|---|
|No| μ | Debye |0.0285|800|
|Yes| μ | Debye | **0.0261**|700|

## Notes

* RAdam is used as an optimizer, so that no warmup is used.
* EMA model is used for evaluation, so that no annealing is used.
* Is should be evaluated with full cross validations with 4 or 5 folds, but I used only 1/4 folds because of no time.

