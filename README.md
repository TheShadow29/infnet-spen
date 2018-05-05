# Inference-SPEN
Course Project for CS726 Advanced Machine Learning

## Initial Setup
1. First prepare the dataset figment. http://cistern.cis.lmu.de/figment/
2. Download entity dataset, entity embeddings (around 2gb) into data/figment. Be sure to unzip entity dataset.
3. `python data/preprocess_figment.py`

1. Prepare the bibtex dataset. http://mulan.sourceforge.net/datasets.html
2. `python data/preprocess_bibtex.py`

1. Download the bookmarks dataset from [here](https://drive.google.com/drive/folders/1dEKnx0d0dgHSdy9OWuqjErrOJCQf1oVl?usp=sharing).
2. Place it in `data/bookmarks`. There is no need to run any proprocessing script.

## Running
1. `python -m mains.infnet --config configs/figment.json`
2. `python -m mains.infnet --config configs/bibtex.json`
3. `python -m mains.infnet --config configs/bookmarks.json`

## Tuning
1. `python generate_configs.py`
2. `mkdir logs`
3. `python run.py`
4. `python analysis.py`

## TODO
- [x] BibTex
- [x] Figment Dataset
- [x] WGAN / Improved WGAN

## Code Description
1. `base/` : Contains the base model and base trainer. The model and trainer are  inherited from here.
2. `configs/` : Contains the configuration files stored in json format. All hyper-parameters are stored here.
3. `data/`:  Contains scripts to process the data files and store them into pickle format
4. `data_loader/`: Contains the class DataGenerator which is used to get data from the pipeline. Since most of our models are small, a naive implementation was fine. In case of bigger datasets, it might be worth looking into the tensorflow dataset api.
5. `mains/` : Contains the main file to be called which is `infnet.py` This takes in the configuration file, and uses it to initiliaze which model, trainer, hyper-parameteres to choose, which parameters to save for tensorboard etc.
5. `models/` : Contains model definitions, each of which is a class. There are 4 such classes. EnergyNet, InferenceNet, FeatureNet, Spen. The first three are simple feed forward networks, and the last one is the actual model which is used and combines all the different networks together.
6. `trainers/` : Contains the trainer, which schedules the training, evaluation, tensorboard logging among different things.
7. `utils/` : Contains utility function like the process_config which is used to parse the configuration file.
8. `analysis.py`, `generate_configs.py`, `run.py`: are all used for  hyper-parameter tuning.

## Acknowledgements:
A major thanks to Lifu Tu and Kevin Gimpel (authors of the paper we have implemented) for sharing their theano code and responding promptly to our queries on the paper. We Lifu for sharing his Theano Code. We also thank David Belanger for the Bookmarks dataset and his original SPEN [implementation](https://github.com/davidBelanger/SPEN).

## References:
Lifu Tu and Kevin Gimpel. Learning approximate inference networks for structured prediction.
CoRR, abs/1803.03376, 2018. URL http://arxiv.org/abs/1803.03376
