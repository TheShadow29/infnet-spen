import tensorflow as tf
import numpy as np
import data_loader.data_generator as data_generator
from data_loader.data_generator import load_embeddings, load_vocab
from models.spen import SPEN
from trainers.spen_trainer import SpenTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import get_logger, TFLogger
from utils.utils import get_args


logger = get_logger(__name__)


def main():
    args = get_args()

    # Set random seeds
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # config is of type Munch
    config = process_config(args.config)

    create_dirs([config.summary_dir, config.checkpoint_dir])

    sconfig = tf.ConfigProto()
    sconfig.gpu_options.allow_growth = True

    sess = tf.Session(config=sconfig)

    with tf.variable_scope("model"):
        model = SPEN(config)

    if config.data.embeddings is True:
        # This is outside data generator since it's used to explicitly init TF model
        embeddings = load_embeddings(config)
        logger.info("embeddings loaded :- %d items", len(embeddings))
    else:
        embeddings = None

    if config.data.vocab is True:
        # Load the two vocabulary files for types and entities
        types, types_vocab, entities, entitites_vocab = load_vocab(config)
        logger.info("vocab loaded :- %d types, %d entities", len(types), len(entities))

    generator = eval("data_generator.%s" % config.data.data_generator)

    dsplits = config.data.splits
    train_data = generator(config, split=dsplits[0])
    logger.info("training set loaded :- %d instances", train_data.len)
    dev_data = generator(config, split=dsplits[1])
    logger.info("dev set loaded :- %d instances", dev_data.len)
    test_data = generator(config, split=dsplits[2])
    logger.info("test set loaded :- %d instances", test_data.len)

    # create tensorboard logger
    tf_logger = TFLogger(sess, config)
    # create trainer and path all previous components to it
    trainer = SpenTrainer(
        sess, model, [train_data, dev_data, test_data], embeddings, config, tf_logger
    )
    # Inference Net pre-training
    trainer.train(stage=0)
    # This is needed to keep a copy of the pre-trained infnet
    trainer.copy_infnet()
    # Energy Network Minimization
    trainer.train(stage=1)
    # Inference Network post-training
    trainer.train(stage=2)


if __name__ == '__main__':
    main()
