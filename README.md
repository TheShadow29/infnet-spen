# Inference-SPEN
Course Project for CS726 Advanced Machine Learning

## Initial Setup
1. First prepare the dataset figment. http://cistern.cis.lmu.de/figment/
2. Download entity dataset, entity embeddings (around 2gb) into data/figment. Be sure to unzip entity dataset.
3. `python data/preprocess_figment.py`

## Running
1. `python -m mains.infnet --config configs/figment.json`