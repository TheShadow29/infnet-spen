import tensorflow as tf
from utils.logger import get_logger

logger = get_logger(__name__)


class BaseModel:
    def __init__(self, config):
        self.config = config
        # init the global step
        self.init_global_step()
        # init the epoch counter
        self.init_cur_epoch()

    # save function thet save the checkpoint in the path defined in configfile
    def save(self, sess):
        self.saver.save(sess, self.config.checkpoint_dir, self.global_step_tensor)

    # load lateset checkpoint from the experiment path defined in config_file
    def load(self, sess):
        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        if latest_checkpoint:
            logger.info("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            logger.info("Model loaded")

    # just inialize a tensorflow variable to use it as epoch counter
    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = \
                [tf.Variable(0, trainable=False, name='cur_epoch_%d' % i) for i in range(3)]
            self.increment_cur_epoch_tensor = [
                tf.assign(self.cur_epoch_tensor[i], self.cur_epoch_tensor[i] + 1)
                for i in range(3)
            ]

    # just inialize a tensorflow variable to use it as global step counter
    def init_global_step(self):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
            self.global_step_tensor_inf = tf.Variable(0, trainable=False, name='global_step_inf')

    def init_saver(self):
        # just copy the following line in your child class
        # self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError
