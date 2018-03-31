from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
import pdb
import tensorflow as tf


class SpenTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super().__init__(sess, model, data, config, logger)

    def train_epoch(self):
        loop = tqdm(range(self.config.train.num_iter_per_epoch))
        losses = []
        accs = []
        for it in loop:
            loss, acc = self.train_step()
            losses.append(loss)
            accs.append(acc)
        loss = np.mean(losses)
        acc = np.mean(accs)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {}
        summaries_dict['loss'] = loss
        summaries_dict['acc'] = acc
        # self.logger.info("Loss %f, Acc %f", loss, acc)
        print('Loss', loss, 'Acc', acc)
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
        _, _, _, loss = self.sess.run([self.model.phi_opt, self.model.theta_opt,
                                       self.model.shi_opt,
                                       self.model.cost],
                                      feed_dict=feed_dict)

        # _, acc = self.sess.run([self.model.shi_opt, self.model.test_cost], feed_dict=feed_dict)
        # try:
        acc = self.sess.run([self.model.acc], feed_dict=feed_dict)
        # assert acc >= 0
        # except Exception as e:
        # pdb.set_trace()
        # acc = self.train_inf_step()
        # pdb.set_trace()
        # assert (loss <= 0).all()

        return loss, acc

    # def train_inf_step(self):
    #     tot_x, tot_y = self.data.data_x, self.data.data_y
    #     feed_dict = {self.model.input_x: tot_x, self.model.labels_y: tot_y,
    #                  self.model.is_training: True}
    #     self.test_cost = tf.reduce_sum(self.model.energy_net3.energy_out)
    #     acc = self.sess.run([self.test_cost], feed_dict=feed_dict)
    #     return acc
