import tensorflow as tf
import sys

from base.base_model import BaseModel
from models.energy_net import EnergyNet
from models.inference_net import InferenceNet
from models.feature_net import FeatureNet
from utils.logger import get_logger

logger = get_logger(__name__)

BASE_ENERGY_NET_SCOPE = "energy_net"
BASE_INFNET_SCOPE = "inference_net"
BASE_COPY_INFNET_SCOPE = "copy_inference_net"

ENERGY_NET_SCOPE = "model/%s" % BASE_ENERGY_NET_SCOPE
INFNET_SCOPE = "model/%s" % BASE_INFNET_SCOPE
COPY_INFNET_SCOPE = "model/%s" % BASE_COPY_INFNET_SCOPE

EPSILON = 1e-7


def random_uniform(limit):
    return tf.random_uniform_initializer(-limit, limit)


class SPEN(BaseModel):
    def __init__(self, config):
        super(SPEN, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        config = self.config
        batch_size = config.train.batch_size

        self.input_x = tf.placeholder(
            tf.int64, shape=[batch_size], name='input_x'
        )
        self.labels_y = tf.placeholder(
            tf.float32, shape=[batch_size, config.type_vocab_size], name='labels_y'
        )

        if config.data.embeddings is True:
            self.build_embeddings_graph()
        else:
            self.build_feature_net()
        # Inference Network and Energy Network
        self.build_subnets()
        self.copy_infnet()
        self.regularize()
        if self.config.ssvm.enable is True:
            self.ssvm_losses()
        else:
            self.adversarial_losses()
        # Inference Network energy minimization & Inference Network pre-training
        self.infnet_losses()
        self.evaluate()

    def build_embeddings_graph(self):
        config = self.config
        vocab_size = config.entities_vocab_size
        e_size = config.feature_size
        # Logic for embeddings
        with tf.name_scope('embeddings'):
            self.embeddings_placeholder = tf.placeholder(
                tf.float32, [vocab_size, e_size], name="embeddings_placeholder"
            )
            self.embeddings = tf.get_variable(
                "embedding", [vocab_size, e_size],
                initializer=random_uniform(0.25),
                trainable=config.embeddings_tune
            )
            self.feature_input = tf.nn.embedding_lookup(self.embeddings, self.input_x)

            # Used in the static / non-static configurations
            self.load_embeddings = self.embeddings.assign(self.embeddings_placeholder)

    def build_feature_net(self):
        self.feature_input = FeatureNet(self.config, self.input_x)

    def build_subnets(self):
        config = self.config
        regularizer = tf.contrib.layers.l2_regularizer(1.0)

        # This is the cost-augmented inference network
        # This is used for GAN-style training
        # After training the energy network, we train the psi parameters
        with tf.variable_scope(BASE_INFNET_SCOPE, regularizer=regularizer):
            self.inference_net = InferenceNet(
                config, self.feature_input
            )
        # Energy network definitions
        with tf.variable_scope(BASE_ENERGY_NET_SCOPE, regularizer=regularizer):
            self.energy_net1 = EnergyNet(
                config, self.feature_input, self.labels_y
            )
            self.red_energy_gt_out = tf.reduce_sum(self.energy_net1.energy_out)
        with tf.variable_scope(BASE_ENERGY_NET_SCOPE, reuse=True, regularizer=regularizer):
            self.energy_net2 = EnergyNet(
                config, self.feature_input, self.inference_net.layer2_out
            )
            self.red_energy_inf_out = tf.reduce_sum(self.energy_net2.energy_out)

    def ssvm_losses(self):
        config = self.config
        regularizer = tf.contrib.layers.l2_regularizer(1.0)
        batch_size = config.train.batch_size
        lamb_reg_theta = config.train.lamb_reg_theta

        # We are going to optimize over this variable
        with tf.variable_scope('ssvm_y'):
            self.ssvm_y_pred = tf.get_variable(
                'ssvm_y_pred', [batch_size, config.type_vocab_size],
                initializer=tf.random_uniform_initializer(0, 1)
            )
            self.ssvm_y_pred_clip = tf.clip_by_value(
                self.ssvm_y_pred, EPSILON, 1 - EPSILON
            )
        # Reusing the Energy function module for this y_pred
        with tf.variable_scope(BASE_ENERGY_NET_SCOPE, reuse=True, regularizer=regularizer):
            self.energy_net3 = EnergyNet(
                config, self.feature_input, self.ssvm_y_pred_clip
            )
            self.red_energy_y_pred = tf.reduce_sum(self.energy_net3.energy_out)

        # Choosing the appropriate difference function
        if config.train.diff_type == 'sq_diff':
            self.ssvm_difference = tf.reduce_sum(
                tf.square(self.labels_y - self.ssvm_y_pred_clip), axis=1
            )
        elif config.train.diff_type == 'abs_diff':
            self.ssvm_difference = tf.reduce_sum(
                tf.abs(self.labels_y - self.ssvm_y_pred_clip), axis=1
            )
        elif config.train.diff_type == 'perceptron':
            self.ssvm_difference = tf.constant([0] * batch_size, dtype=tf.float32)
        elif config.train.diff_type == 'slack':
            self.ssvm_difference = tf.constant([1] * batch_size, dtype=tf.float32)
        self.ssvm_red_difference = tf.reduce_sum(self.ssvm_difference)

        # Defining the SSVM loss criterion
        ssvm_max_difference = tf.maximum(
            self.ssvm_difference - self.energy_net3.energy_out + self.energy_net1.energy_out,
            tf.constant([0] * batch_size, dtype=tf.float32)
        )
        self.ssvm_base_objective = tf.reduce_sum(ssvm_max_difference)
        # Defining the loss augmented inference loss
        self.cost_augmented_inference = tf.reduce_sum(
            -1 * self.ssvm_difference + self.energy_net3.energy_out
        )
        # This loss term is just used for inference by directly minimizing energy
        self.cost_inference = tf.reduce_sum(
            self.energy_net3.energy_out
        )
        # Defining the loss function for theta
        self.ssvm_cost_theta = \
            self.ssvm_base_objective + \
            lamb_reg_theta * self.reg_losses_theta
        # Defining all the optimizers
        # Defining the optimizer for y, inner loop of optimization
        self.ssvm_y_opt = tf.train.GradientDescentOptimizer(config.ssvm.lr_inference).minimize(
            self.cost_augmented_inference, var_list=[self.ssvm_y_pred]
        )
        # This optimizer just used during inference
        self.ssvm_infer_y_opt = tf.train.GradientDescentOptimizer(config.ssvm.lr_inference).minimize(
            self.cost_inference, var_list=[self.ssvm_y_pred]
        )
        # Defining the optimizer for theta variables during training
        theta_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=ENERGY_NET_SCOPE)
        self.ssvm_theta_opt = tf.train.AdamOptimizer(config.train.lr_theta).minimize(
            self.ssvm_cost_theta, global_step=self.global_step_tensor, var_list=theta_vars
        )

    def copy_infnet(self):
        config = self.config
        regularizer = tf.contrib.layers.l2_regularizer(1.0)
        # These variables will store values of a preprocessed inference net
        # This allows us to keep a copy of the pre-trained network
        # Which can be used as a regularization term Section 5, Tu & Gimpel 2018
        with tf.variable_scope(BASE_COPY_INFNET_SCOPE, regularizer=regularizer):
            self.copy_inference_net = InferenceNet(
                config, self.feature_input
            )
        copy_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=COPY_INFNET_SCOPE)
        infnet_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=INFNET_SCOPE)
        # As a sanity check, ensure that the copy_vars and infnet_vars are in the same order
        for copy_var, infnet_var in zip(copy_vars, infnet_vars):
            copy_var_name = copy_var.name[len(COPY_INFNET_SCOPE):]
            infnet_var_name = infnet_var.name[len(INFNET_SCOPE):]
            if copy_var_name != infnet_var_name:
                logger.error("Variable name order mismatch, exiting")
                sys.exit(0)
        # Create assignment operators to initialize pre_inference_net
        with tf.name_scope('copy_infnet'):
            self.copy_infnet_ops = [
                copy_var.assign(infnet_var)
                for copy_var, infnet_var in zip(copy_vars, infnet_vars)
            ]
            # The last loss term in Section 5, Tu & Gimpel 2018
            self.var_diff = [
                tf.reduce_sum(tf.square(copy_var - infnet_var))
                for copy_var, infnet_var in zip(copy_vars, infnet_vars)
            ]
            self.pre_train_bias = tf.add_n(self.var_diff)

    def regularize(self):
        with tf.name_scope('regularize'):
            self.reg_losses_phi = tf.add_n(
                tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=INFNET_SCOPE)
            )
            self.reg_losses_theta = tf.add_n(
                tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=ENERGY_NET_SCOPE)
            )
            # Entropy Regularization, Section 5 of Tu & Gimpel 2018
            # We add the EPSILON constant for numerical stability
            prob = tf.clip_by_value(self.inference_net.layer2_out, EPSILON, 1 - EPSILON)
            not_prob = 1 - prob
            self.reg_losses_entropy = tf.reduce_sum(
                -1 * prob * tf.log(prob) - not_prob * tf.log(not_prob)
            )

    def adversarial_losses(self):
        config = self.config
        batch_size = config.train.batch_size

        lamb_reg_phi = config.train.lamb_reg_phi
        lamb_reg_theta = config.train.lamb_reg_theta
        lamb_reg_entropy = config.train.lamb_reg_entropy
        lamb_pretrain_bias = config.train.lamb_pretrain_bias

        with tf.name_scope('base_objective'):
            if config.train.diff_type == 'sq_diff':
                self.difference = tf.reduce_sum(
                    tf.square(self.labels_y - self.inference_net.layer2_out), axis=1
                )
            elif config.train.diff_type == 'abs_diff':
                self.difference = tf.reduce_sum(
                    tf.abs(self.labels_y - self.inference_net.layer2_out), axis=1
                )
            elif config.train.diff_type == 'perceptron':
                self.difference = tf.constant([0] * batch_size, dtype=tf.float32)
            elif config.train.diff_type == 'slack':
                self.difference = tf.constant([1] * batch_size, dtype=tf.float32)
            self.red_difference = tf.reduce_sum(self.difference)

            # Applying the hinge to the loss function
            max_difference = tf.maximum(
                self.difference - self.energy_net2.energy_out + self.energy_net1.energy_out,
                tf.constant([0] * batch_size, dtype=tf.float32)
            )
            self.base_objective_real = tf.reduce_sum(
                self.difference - self.energy_net2.energy_out + self.energy_net1.energy_out
            )
            self.base_objective = tf.reduce_sum(max_difference)

        with tf.name_scope('theta_cost'):
            self.cost_theta = \
                self.base_objective + \
                lamb_reg_theta * self.reg_losses_theta

        with tf.name_scope('gain_phi'):
            # Negative sign for reg_losses_phi since we want to maximize phi objective
            self.gain_phi = \
                self.base_objective - \
                lamb_reg_phi * self.reg_losses_phi - \
                lamb_reg_entropy * self.reg_losses_entropy - \
                lamb_pretrain_bias * self.pre_train_bias
            self.cost_phi = -1 * self.gain_phi

        with tf.name_scope('phi_opt'):
            phi_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=INFNET_SCOPE)
            self.phi_opt = tf.train.AdamOptimizer(config.train.lr_phi).minimize(
                self.cost_phi, global_step=self.global_step_tensor, var_list=phi_vars
            )
        with tf.name_scope('theta_opt'):
            theta_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=ENERGY_NET_SCOPE)
            self.theta_opt = tf.train.AdamOptimizer(config.train.lr_theta).minimize(
                self.cost_theta, var_list=theta_vars
            )

    def infnet_losses(self):
        with tf.name_scope('infnet_losses'):
            infnet_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=INFNET_SCOPE)
            # Minimizing a cross entropy loss for independent classes
            prob = tf.clip_by_value(self.inference_net.layer2_out, EPSILON, 1 - EPSILON)
            not_prob = 1 - prob
            self.infnet_ce_loss = tf.reduce_sum(
                -1 * (1 - self.labels_y) * tf.log(not_prob) - self.labels_y * tf.log(prob)
            )
            self.infnet_ce_opt = tf.train.AdamOptimizer(self.config.train.lr_psi).minimize(
                self.infnet_ce_loss, var_list=infnet_vars
            )

        with tf.name_scope('psi_cost'):
            # Psi parameter optimization, Equation (5) in Tu & Gimpel, 2018
            self.energy_psi = tf.reduce_sum(self.energy_net2.energy_out)

        with tf.name_scope('psi_opt'):
            self.psi_opt = tf.train.AdamOptimizer(self.config.train.lr_psi).minimize(
                self.energy_psi, var_list=infnet_vars
            )

    def evaluate(self):
        with tf.name_scope('evaluate'):
            self.probabilities = self.inference_net.layer2_out
            # infnet based inference
            self.outputs = tf.round(self.probabilities)
            self.diff = tf.cast(
                tf.reduce_sum(tf.abs(self.outputs - self.labels_y), axis=1),
                dtype=tf.int64
            )
            if self.config.ssvm.enable is True:
                # SSVM based inference
                # Please optimize over ssvm_y_pred before using these nodes
                self.ssvm_outputs = tf.round(self.ssvm_y_pred_clip)
                self.ssvm_diff = tf.cast(
                    tf.reduce_sum(tf.abs(self.ssvm_outputs - self.labels_y), axis=1),
                    dtype=tf.int64
                )
            self.results = self.config.train.batch_size - tf.count_nonzero(self.diff)

    def init_saver(self):
        with tf.name_scope('saver'):
            # here you initalize the tensorflow saver that will be used in saving the checkpoints.
            self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
