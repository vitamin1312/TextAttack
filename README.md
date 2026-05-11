# Text Attack experiment with adversarial augmentations

This code is mainly adapted to run in Google Colab or Kaggle with preinstalled dependencies

# Installing
To install dependencies run [`install.sh`](./install.sh)

# Experiments
To run all experiments use [`run.sh`](./run.sh). This script deletes directories. Use with caution. If you want to run experiments manually, you can use code from `run.sh`. In general, [`train.py`](./train.py) is used to train the model, [`attack.py`](./attack.py) to attack for evaluation.

To collect results use [`collect.py`](./collect.py)

# Models configuration and training

# Model
| Parameter | Value |
|----------|----------|
| architectures | LSTMForClassification |
| hidden_size | 150 |
| depth | 1 |
| dropout | 0.3 |
| num_labels | 2 |
| max_seq_length | 128 |
| emb_layer_trainable | true |

# Training config
| Parameter | Value |
|-----------|-------|
| SEED | 42 |
| N_TRAIN | 3000 |
| N_EVAL | 872 |
| lr | 3e-4 |
| batch | 32 |
| eval-batch | 32 |

# Results
| Model / Attack | - | bae | deepwordbug | hotflip | pruthi | textfooler |
|----------------|----|-----|-------------|---------|--------|-------------|
| baseline | 77.06 | 24.54 | 22.59 | 75.8 | 60.32 | 3.56 |
| deepwordbug20 | 74.66 | 18.58 | 21.1 | 72.94 | 52.98 | 1.15 |
| deepwordbug75 | 67.66 | 19.72 | 30.5 | 65.83 | 49.54 | 1.38 |
| textfooler20 | 73.51 | 23.62 | 21.79 | 54.36 | 52.06 | 15.02 |

# Additional information
[Pretrained models](https://drive.google.com/drive/u/1/folders/1tlEHfayVU0AjjvCmlgKu1JScYaNuSGdT)

[Research info](https://drive.google.com/drive/folders/1F7QSLeM4z4f21RMk-iDrsRX-GUp7oUr4?usp=drive_link)

