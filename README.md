# Learning Approximate Inference Networks for Structured Prediction

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

## Config File Information
1. `exp_name` : Name of the experiment
2. `data`: Contains info about the data
   1. `dataset`: Name of the dataset
   2. `data_dir`: Path for the top directory of data
   3. `splits` : Splits for train, validation, test.
   4. `embeddings`: True/False. True if pre-trained embeddings are available.
   5. `vocab` : Same as above
   6. `data_generator`: Name of the data generator defined in `data_loader.py`.
3. `tensorboard_train`: Set True to save tensorboard  in the stage2
4. `tensorboard_infer`: Same as above in stage3
5. `feature_size`: Size of hidden layer in Feature Network
6. `label_measurements`: Same as above in Energy Network
7. `type_vocab_size`: Number of output labels.
8. `entities_vocab_size`: Lookup table for embeddings.
9. `embeddings_tune`: Set to true if embeddings vector to be updated.
10. `max_to_keep`: Requiredby tensorflow saver that will be used in saving the checkpoints.
11. `num_epochs` : Number of epochs in each stage
12. `train`: Info about how to train
    1. `diff_type`: \nabla operator in the paper
    2. `batch_size`: batch size for training
    3. `state_size`: Not required. Kept for historical reasons.
    4. `hidden_units`: Hidden units in inference and feature net (depends on embeddings is true or false)
    5. `lr_*`: learning rate for optimization of corresponding variable
    6. `lambda_*`: lambda regularization for optimization of corresponding variable
    7. `lambda_pretrain_bias`: How much to weigh the pretrained network (another term in paper).
    8. `wgan_mode`: Improved WGAN penalty or not.
    9. `lamb_wgan`: regularization for wgan penalty.
13. `ssvm`: Implementation of SPEN 2016 by Belanger. Not complete since we couldn't find implementation of entropic gradient descent.
    1. `enable`: To be ssvm or not to be
    2. `steps`: Number of optimization steps in ssvm
    3. `eval`: True if ssvm inference to be used.
    4. `lr_inference`: learning rate for ssvm inference
14. `eval_print`: What all to print for evaluation
    1. `f1`: f1 score
    2. `accuracy`: accuracy
    3. `energy`: energy
    4. `pretrain`: energy / loss of pretrain
    5. `infnet`: energy / loss of inference network
    6. `f1_score_mode`: Set to examples to compute F1 score averaged over examples. Do label for F1 score averaged over labels. The paper does it over examples.
    7. `threshold`: Threshold adjusted on validation set
    8. `time_taken`: For time evaluation. Only training/inference step time. Not whole time.

## Acknowledgements:
A major thanks to Lifu Tu and Kevin Gimpel (authors of the paper we have implemented) for sharing their Theano code and responding promptly to our queries on the paper. We thank Lifu for sharing his Theano Code. We also thank David Belanger for the Bookmarks dataset and his original SPEN [implementation](https://github.com/davidBelanger/SPEN).

## References:
Lifu Tu and Kevin Gimpel. Learning approximate inference networks for structured prediction.
CoRR, abs/1803.03376, 2018. URL http://arxiv.org/abs/1803.03376
