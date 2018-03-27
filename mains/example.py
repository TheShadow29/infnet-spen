import tensorflow as tf
import os
from data_loader.data_generator import figment_data_generator
from models.example_model import ExampleModel
from trainers.example_trainer import ExampleTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args
import pdb


def main():
    # capture the config path from the run arguments
    # then process the json configration file
    try:
        args = get_args()
        # pdb.set_trace()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    # create tensorflow session
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    sconfig = tf.ConfigProto()
    sconfig.gpu_options.allow_growth = True
    sconfig.gpu_options.per_process_gpu_memory_fraction = 0.4
    sess = tf.Session(config=sconfig)
    # create instance of the model you want
    model = ExampleModel(config)
    # load model if exist
    model.load(sess)
    # create your data generator
    # data = DataGenerator(config)
    data = figment_data_generator(config)
    # create tensorboard logger
    logger = Logger(sess, config)
    # create trainer and path all previous components to it
    trainer = ExampleTrainer(sess, model, data, config, logger)

    # here you train your model
    trainer.train()


if __name__ == '__main__':
    main()
