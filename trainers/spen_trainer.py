from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
import sys
from utils.logger import get_logger


logger = get_logger(__name__)
EPSILON = 1e-7


class SpenTrainer(BaseTrain):
    def __init__(self, sess, model, data, embeddings, config, tf_logger):
        super().__init__(sess, model, data, config, tf_logger)

        if self.config.data.embeddings is True:
            # Push these embeddings into TensorFlow graph
            feed_dict = {
                model.embeddings_placeholder.name: embeddings
            }
            sess.run(model.load_embeddings, feed_dict=feed_dict)

        # Housekeeping tasks for training
        self.train_data, self.dev_data, self.test_data = data
        self.num_batches_train = int(
            np.ceil(self.train_data.len / self.config.train.batch_size)
        )
        self.num_batches_dev = int(
            np.ceil(self.dev_data.len / self.config.train.batch_size)
        )

    def train_epoch(self, cur_epoch, stage=0):
        if stage == 0:
            self.current_corpus = self.train_data
            for batch in tqdm(range(self.num_batches_train)):
                self.batch_num = batch
                # Inference Net pre-training
                self.step_pretrain()
        elif stage == 1:
            self.current_corpus = self.train_data
            for batch in tqdm(range(self.num_batches_train)):
                # Energy Network Minimization
                if self.config.ssvm.enable is True:
                    self.step_ssvm()
                else:
                    self.step_adversarial()
        else:
            self.current_corpus = self.train_data
            for batch in tqdm(range(self.num_batches_train)):
                self.batch_num = batch
                # Inference Network post-training
                self.step_infnet_energy()

        # Verify that the data is completed before moving to evaluation
        for corpus in [self.train_data, self.dev_data, self.test_data]:
            if corpus.batch_pointer != 0:
                logger.info("corpus train not completed")
                sys.exit(0)

        self.evaluate()
        self.model.save(self.sess)
        logger.info("Completed epoch %d / %d", cur_epoch + 1, self.config.num_epochs[stage])

    def get_feed_dict(self):
        corpus = self.current_corpus
        batch_size = self.config.train.batch_size

        batch_x, batch_y = next(corpus.next_batch(batch_size))
        if len(batch_x) < batch_size:
            batch_x, batch_y = self.pad_batch(batch_x, batch_y)
        feed_dict = {
            self.model.input_x: batch_x,
            self.model.labels_y: batch_y
        }
        return feed_dict

    def step_pretrain(self):
        self.sess.run(self.model.feat_ce_opt, feed_dict=self.get_feed_dict())

    def copy_infnet(self):
        logger.info("Copying trained inference net weights")
        self.sess.run(self.model.copy_infnet_ops)
        self.evaluate()

    def step_adversarial(self):
        feed_dict = self.get_feed_dict()
        self.sess.run(self.model.phi_opt, feed_dict=feed_dict)
        self.sess.run(self.model.theta_opt, feed_dict=feed_dict)
        if self.config.tensorboard is True:
            self.summaries = {
                'base_objective': self.model.base_objective,
                'base_obj_real': self.model.base_objective_real,
                'energy_inf_net': self.model.red_energy_inf_out,
                'energy_ground_truth': self.model.red_energy_gt_out,
                'margin_loss': self.model.red_difference,
                'reg_losses_theta': self.model.reg_losses_theta,
                'reg_losses_phi': self.model.reg_losses_phi,
                'reg_losses_entropy': self.model.reg_losses_entropy,
                'pretrain_bias': self.model.pretrain_bias
            }
            self.tf_logger.summarize(
                self.model.global_step_tensor.eval(self.sess),
                summaries_dict=self.sess.run(self.summaries, feed_dict)
            )

    def step_ssvm(self):
        feed_dict = self.get_feed_dict()
        # begin by initializing self.ssvm_y_pred to random value
        self.sess.run(self.model.ssvm_y_pred.initializer)
        # inner optimization loop over y
        for i in range(self.config.ssvm.steps):
            self.sess.run(self.model.ssvm_y_opt, feed_dict=feed_dict)
        # with max-margin y, run a theta update
        self.sess.run(self.model.ssvm_theta_opt, feed_dict=feed_dict)
        if self.config.tensorboard is True:
            self.summaries = {
                'base_objective': self.model.ssvm_base_objective,
                'energy_y_pred': self.model.red_energy_y_pred,
                'energy_ground_truth': self.model.red_energy_gt_out,
                'ssvm_difference': self.model.ssvm_red_difference,
                'reg_losses_theta': self.model.reg_losses_theta
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
        if self.config.data.embeddings is False:
            # Input is a 2-D array
            extension_x = np.tile(batch_x[-1], (batch_size - total, 1))
            new_batch_x = np.concatenate((batch_x, extension_x), axis=0)
        else:
            # Input is a 1-D array
            extension_x = np.tile(batch_x[-1], batch_size - total)
            new_batch_x = np.concatenate((batch_x, extension_x), axis=0)
        extension_y = np.tile(batch_y[-1], (batch_size - total, 1))
        new_batch_y = np.concatenate((batch_y, extension_y), axis=0)

        return new_batch_x, new_batch_y

    def evaluate(self):
        config = self.config

        batch_size = self.config.train.batch_size
        inference_mode = "ssvm" if config.ssvm.eval is True else "inference"
        f1_score_function = \
            self.f1_score_examples if config.eval_print.f1_score_mode == 'examples' else self.f1_score_labels

        infnet_threshold_f1 = -1
        infnet_threshold_acc = -1

        # finding performance on both dev and test dataset
        for corpus in [self.dev_data, self.test_data]:
            total_energy = 0.0

            all_pretrain_probs = np.zeros((corpus.len, config.type_vocab_size))
            all_infnet_probs = np.zeros((corpus.len, config.type_vocab_size))
            all_gt_outputs = np.zeros((corpus.len, config.type_vocab_size))

            num_batches = int(np.ceil(corpus.len / batch_size))
            for batch in range(num_batches):
                batch_x, batch_y = next(corpus.next_batch(batch_size))
                total = len(batch_x)
                if len(batch_x) < batch_size:
                    batch_x, batch_y = self.pad_batch(batch_x, batch_y)

                # Enter code to calculate accuracy here
                feed_dict = {
                    self.model.input_x: batch_x,
                    self.model.labels_y: batch_y
                }

                if self.config.ssvm.eval is True:
                    # Use the ssvm inference objective
                    # begin by initializing self.ssvm_y_pred to random value
                    self.sess.run(self.model.ssvm_y_pred.initializer)
                    # inner optimization loop over y
                    for i in range(config.ssvm.steps):
                        self.sess.run(self.model.ssvm_infer_y_opt, feed_dict=feed_dict)
                    outputs = [
                        self.model.energy_net1.energy_out,
                        self.model.pretrain_probs,
                        self.model.ssvm_probs
                    ]
                else:
                    outputs = [
                        self.model.energy_net1.energy_out,
                        self.model.pretrain_probs,
                        self.model.infnet_probs
                    ]

                energy, pretrain_probs, infnet_probs = self.sess.run(outputs, feed_dict=feed_dict)
                total_energy += np.sum(energy[:total])

                all_pretrain_probs[batch * batch_size:(batch + 1) * batch_size] = pretrain_probs[:total]
                all_infnet_probs[batch * batch_size:(batch + 1) * batch_size] = infnet_probs[:total]
                all_gt_outputs[batch * batch_size:(batch + 1) * batch_size] = batch_y[:total]

            # Verify that the data is completed
            if corpus.batch_pointer != 0:
                logger.info("corpus %s not completed" % corpus.split)
                sys.exit(0)

            if config.eval_print.energy is True:
                logger.info("Ground truth energy on %s corpus is %.4f", corpus.split, total_energy)

            if config.eval_print.f1 is True:
                if config.eval_print.pretrain is True:
                    pretrain_f1_score = f1_score_function(all_pretrain_probs, all_gt_outputs, 0.5)
                    logger.info("F1 score of pretrained network on %s is %.4f", corpus.split, pretrain_f1_score)

                if config.eval_print.infnet is True:
                    if infnet_threshold_f1 == -1:
                        thresholds = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.15, 0.2, 0.25, 0.30,
                                      0.35, 0.4, 0.45, 0.5, 0.55, 0.60, 0.65, 0.70, 0.75]
                        best_f1 = -1.0
                        for thresh in thresholds:
                            f1_score = f1_score_function(all_infnet_probs, all_gt_outputs, thresh)
                            infnet_threshold_f1 = thresh if f1_score > best_f1 else infnet_threshold_f1
                            best_f1 = f1_score if f1_score > best_f1 else best_f1
                        logger.info("Chosen threshold value for f1 is %.4f", infnet_threshold_f1)
                    else:
                        best_f1 = f1_score_function(all_infnet_probs, all_gt_outputs, infnet_threshold_f1)

                    logger.info("F1 score of %s network on %s is %.4f", inference_mode, corpus.split, best_f1)

            if config.eval_print.accuracy is True:
                if config.eval_print.pretrain is True:
                    total_pretrain_correct = self.accuracy(all_pretrain_probs, all_gt_outputs, 0.5)
                    logger.info(
                        "Accuracy of pretrained network on %s is %.4f (%d / %d)",
                        corpus.split, total_pretrain_correct / corpus.len, total_pretrain_correct, corpus.len
                    )

                if config.eval_print.infnet is True:
                    if infnet_threshold_acc == -1:
                        thresholds = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.15, 0.2, 0.25, 0.30,
                                      0.35, 0.4, 0.45, 0.5, 0.55, 0.60, 0.65, 0.70, 0.75]
                        best_infnet_correct = -1.0
                        for thresh in thresholds:
                            infnet_correct = self.accuracy(all_infnet_probs, all_gt_outputs, thresh)
                            infnet_threshold_acc = \
                                thresh if infnet_correct > best_infnet_correct else infnet_threshold_acc
                            best_infnet_correct = \
                                infnet_correct if infnet_correct > best_infnet_correct else best_infnet_correct
                        logger.info("Chosen threshold value for accuracy is %.4f", infnet_threshold_acc)
                    else:
                        best_infnet_correct = self.accuracy(all_infnet_probs, all_gt_outputs, infnet_threshold_acc)

                    logger.info(
                        "Accuracy of %s on %s is %.4f (%d / %d)",
                        inference_mode, corpus.split, best_infnet_correct / corpus.len, best_infnet_correct, corpus.len
                    )

    def accuracy(self, probs, gt_outputs, threshold):
        outputs = np.greater(probs, threshold)
        difference = np.sum(np.abs(outputs - gt_outputs), axis=1)
        correct = np.count_nonzero(np.isclose(difference, 0))
        return correct

    def f1_score_labels(self, probs, gt_outputs, threshold):
        type_vocab_size = self.config.type_vocab_size
        precision_total = 0.0
        recall_total = 0.0

        outputs = np.greater(probs, threshold)

        for i in range(type_vocab_size):
            # Hacky way to allow us to use np.count_nonzero function
            # Labels of type predicted=1, gt=1
            tp = np.count_nonzero(
                np.isclose(2 * outputs[:, i] - gt_outputs[:, i], 1)
            )
            # Labels of type predicted=1, gt=0
            fp = np.count_nonzero(
                np.isclose(outputs[:, i] - gt_outputs[:, i], 1)
            )
            # Labels of type predicted=0, gt=1
            fn = np.count_nonzero(
                np.isclose(outputs[:, i] - gt_outputs[:, i], -1)
            )
            precision_total += float(tp) / (tp + fp + EPSILON)
            recall_total += float(tp) / (tp + fn + EPSILON)
        # Macro averaging of F1 score
        precision = precision_total / type_vocab_size
        recall = recall_total / type_vocab_size
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def f1_score_examples(self, probs, gt_outputs, threshold):
        f1_total = 0.0

        outputs = np.greater(probs, threshold)

        # Hacky way to allow us to use np.count_nonzero function
        # Labels of type predicted=1, gt=1
        tp = np.count_nonzero(
            np.isclose(2 * outputs - gt_outputs, 1), axis=1
        )
        # Labels of type predicted=1, gt=0
        fp = np.count_nonzero(
            np.isclose(outputs - gt_outputs, 1), axis=1
        )
        # Labels of type predicted=0, gt=1
        fn = np.count_nonzero(
            np.isclose(outputs - gt_outputs, -1), axis=1
        )
        precision = tp / (tp + fp + EPSILON)
        recall = tp / (tp + fn + EPSILON)
        f1_total = (2 * precision * recall) / (precision + recall + EPSILON)

        f1 = np.sum(f1_total) / len(outputs)
        return f1
