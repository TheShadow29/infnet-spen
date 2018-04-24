import tensorflow as tf
from utils.logger import get_logger

logger = get_logger(__name__)


class BaseTrain:
    def __init__(self, sess, model, data, config, tf_logger):
        self.model = model
        self.tf_logger = tf_logger
        self.config = config
        self.sess = sess
        self.data = data
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

    def train(self, stage=0):
        logger.info("training on stage %d", stage)
        for cur_epoch in range(self.model.cur_epoch_tensor[stage].eval(self.sess),
                               self.config.num_epochs[stage], 1):
            self.train_epoch(cur_epoch, stage)
            self.sess.run(self.model.increment_cur_epoch_tensor[stage])

    def train_epoch(self):
        """
        implement the logic of epoch:
        -loop ever the number of iteration in the config and call teh train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def train_step(self):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError
