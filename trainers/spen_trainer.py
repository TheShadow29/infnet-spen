from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from utils.logger import get_logger

logger = get_logger(__name__)


class SpenTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super().__init__(sess, model, data, config, logger)

    def train_epoch(self):
        loop = tqdm(range(self.config.train.num_iter_per_epoch))
        losses = []
        for it in loop:
            loss = self.train_step()
            losses.append(loss)
        loss = np.mean(losses)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {}
        summaries_dict['loss'] = loss

        logger.info("Loss function :- %.4f", loss)

        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)

    def train_step(self):
        batch_x, batch_y = next(self.data.next_batch(self.config.train.batch_size))
        feed_dict = {self.model.input_x: batch_x, self.model.labels_y: batch_y,
                     self.model.is_training: True}
        # _, loss, acc = self.sess.run([self.model.phi_opt, self.model.theta_opt,
        #                               self.model.cost,
        #                               self.model.accuracy],
        #                              feed_dict=feed_dict)
        _, loss = self.sess.run([self.model.phi_opt, self.model.base_objective], feed_dict=feed_dict)
        _, loss = self.sess.run([self.model.theta_opt, self.model.base_objective], feed_dict=feed_dict)

        return loss

    # def train_inf_step(self):
    #     tot_x, tot_y = self.data.data_x, self.data.data_y
    #     feed_dict = {self.model.input_x: tot_x, self.model.labels_y: tot_y,
    #                  self.model.is_training: True}
    #     self.test_cost = tf.reduce_sum(self.model.energy_net3.energy_out)
    #     acc = self.sess.run([self.test_cost], feed_dict=feed_dict)
    #     return acc
