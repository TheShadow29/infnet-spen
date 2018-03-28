import tensorflow as tf
from data_loader.data_generator import FigmentDataGenerator, load_embeddings, load_vocab
from models.example_model import SPEN
from trainers.example_trainer import ExampleTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import get_logger, TFLogger
from utils.utils import get_args

import sys


logger = get_logger(__name__)


def main():
    args = get_args()
    config = process_config(args.config)

    create_dirs([config.summary_dir, config.checkpoint_dir])

    sconfig = tf.ConfigProto()
    sconfig.gpu_options.allow_growth = True
    # sconfig.gpu_options.per_process_gpu_memory_fraction = 0.4

    sess = tf.Session(config=sconfig)
    # create instance of the model you want
    # model = ExampleModel(config)
    # # load model if exist
    # model.load(sess)

    # This is outside data generator since it's used to explicitly init TF model
    embeddings = load_embeddings(config)
    logger.info("embeddings loaded :- %d items", len(embeddings))

    # Load the two vocabulary files for types and entities
    types, types_vocab, entities, entitites_vocab = load_vocab(config)
    logger.info("vocab loaded :- %d types, %d entities", len(types), len(entities))

    train_data = FigmentDataGenerator(config, split='Etrain')
    logger.info("training set loaded :- %d instances", train_data.len)
    dev_data = FigmentDataGenerator(config, split='Edev')
    logger.info("dev set loaded :- %d instances", dev_data.len)
    test_data = FigmentDataGenerator(config, split='Etest')
    logger.info("test set loaded :- %d instances", test_data.len)

    # Updating configuration file


    sys.exit(0)

    # create tensorboard logger
    tf_logger = TFLogger(sess, config)
    # create trainer and path all previous components to it
    trainer = ExampleTrainer(sess, model, train_data, config, tf_logger)

    # here you train your model
    trainer.train()


if __name__ == '__main__':
    main()
