from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)


class SpenTrainer(BaseTrain):
    def __init__(self, sess, model, model_eval, data, embeddings, config, tf_logger):
        super().__init__(sess, model, model_eval, data, config, tf_logger)
        # Push these embeddings into TensorFlow graph
        feed_dict = {
            model.embeddings_placeholder.name: embeddings
        }
        sess.run(model.load_embeddings, feed_dict=feed_dict)
        # Housekeeping tasks for training
        self.train_data, self.dev_data, self.test_data = data
        self.num_batches = int(
            np.ceil(self.train_data.len / self.config.train.batch_size)
        )

    def train_epoch(self, cur_epoch, stage=0):
        for batch in tqdm(range(self.num_batches)):
            if stage == 0:
                # Inference Net pre-training
                self.step_infnet_classifier()
            elif stage == 1:
                # Energy Network Minimization
                self.step_energy_net()
            else:
                # Inference Network post-training
                self.step_infnet_energy()
        self.evaluate()
        self.model.save(self.sess)
        logger.info("Completed epoch %d / %d", cur_epoch + 1, self.config.num_epochs[stage])

    def get_feed_dict(self):
        batch_size = self.config.train.batch_size

        batch_x, batch_y = next(self.train_data.next_batch(batch_size))
        if len(batch_x) < batch_size:
            batch_x, batch_y = self.pad_batch(batch_x, batch_y)
        feed_dict = {
            self.model.input_x: batch_x,
            self.model.labels_y: batch_y
        }
        return feed_dict

    def step_infnet_classifier(self):
        self.sess.run(self.model.infnet_ce_opt, feed_dict=self.get_feed_dict())

    def copy_infnet(self):
        logger.info("Copying trained inference net weights")
        self.sess.run(self.model.copy_infnet_ops)

    def step_energy_net(self):
        feed_dict = self.get_feed_dict()
        self.sess.run(self.model.phi_opt, feed_dict=feed_dict)
        self.sess.run(self.model.theta_opt, feed_dict=feed_dict)
        self.summaries = {
            'base_objective': self.model.base_objective,
            'reg_losses_theta': self.model.reg_losses_theta,
            'reg_losses_phi': self.model.reg_losses_phi,
            'reg_losses_entropy': self.model.reg_losses_entropy,
            'pre_train_bias': self.model.pre_train_bias
        }
        self.tf_logger.summarize(
            self.model.global_step_tensor.eval(self.sess),
            summaries_dict=self.sess.run(self.summaries, feed_dict)
        )

    def step_infnet_energy(self):
        self.sess.run(self.model.psi_opt, feed_dict=self.get_feed_dict())

    def pad_batch(self, batch_x, batch_y):
        batch_size = self.config.train.batch_size
        total = len(batch_x)

        extension_x = np.tile(batch_x[-1], batch_size - total)
        new_batch_x = np.concatenate((batch_x, extension_x), axis=0)
        extension_y = np.tile(batch_y[-1], (batch_size - total, 1))
        new_batch_y = np.concatenate((batch_y, extension_y), axis=0)

        return new_batch_x, new_batch_y

    def evaluate(self):
        batch_size = self.config.train.batch_size
        # finding performance on both dev and test dataset
        for corpus in [self.train_data, self.dev_data, self.test_data]:
            total_energy = 0.0
            total_correct = 0
            num_batches = int(np.ceil(corpus.len / batch_size))
            for batch in range(num_batches):
                batch_x, batch_y = next(self.train_data.next_batch(batch_size))
                total = len(batch_x)
                if len(batch_x) < batch_size:
                    batch_x, batch_y = self.pad_batch(batch_x, batch_y)

                # Enter code to calculate accuracy here
                feed_dict = {
                    self.model.input_x: batch_x,
                    self.model.labels_y: batch_y
                }
                outputs = [self.model.energy_net1.energy_out, self.model.diff]
                energy, diff = self.sess.run(outputs, feed_dict=feed_dict)
                total_energy += np.sum(energy[:total])
                total_correct += total - np.count_nonzero(diff[:total])

            logger.info("Ground truth energy on %s corpus is %.4f", corpus.split, total_energy)
            logger.info(
                "Accuracy of inference network is %d / %d = %.4f",
                total_correct, corpus.len, total_correct / corpus.len
            )
