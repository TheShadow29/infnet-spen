import tensorflow as tf
import sys

from base.base_model import BaseModel
from models.energy_net import EnergyNet
from models.inference_net import InferenceNet
from utils.logger import get_logger

logger = get_logger(__name__)

BASE_ENERGY_NET_SCOPE = "energy_net"
BASE_INFNET_SCOPE = "inference_net"
BASE_COPY_INFNET_SCOPE = "copy_inference_net"

ENERGY_NET_SCOPE = "model/%s" % BASE_ENERGY_NET_SCOPE
INFNET_SCOPE = "model/%s" % BASE_INFNET_SCOPE
COPY_INFNET_SCOPE = "model/%s" % BASE_COPY_INFNET_SCOPE


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
            tf.int64, shape=[batch_size]
        )
        self.labels_y = tf.placeholder(
            tf.float32, shape=[batch_size, config.type_vocab_size]
        )

        self.create_embeddings_graph()
        # Inference Network and Energy Network
        self.build_subnets()
        self.copy_infnet()
        self.regularize()
        self.create_losses()
        # Inference Network energy minimization & Inference Network pre-training
        self.infnet_losses()
        self.evaluate()

    def create_embeddings_graph(self):
        config = self.config
        vocab_size = config.entities_vocab_size
        e_size = config.embedding_size
        # Logic for embeddings
        self.embeddings_placeholder = tf.placeholder(
            tf.float32, [vocab_size, e_size]
        )
        self.embeddings = tf.get_variable(
            "embedding", [vocab_size, e_size],
            initializer=random_uniform(0.25),
            trainable=config.embeddings_tune
        )
        self.feature_input = tf.nn.embedding_lookup(self.embeddings, self.input_x)

        # Used in the static / non-static configurations
        self.load_embeddings = self.embeddings.assign(self.embeddings_placeholder)

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
        with tf.variable_scope(BASE_ENERGY_NET_SCOPE, reuse=True, regularizer=regularizer):
            self.energy_net2 = EnergyNet(
                config, self.feature_input, self.inference_net.layer2_out
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
        self.reg_losses_phi = tf.add_n(
            tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=INFNET_SCOPE)
        )
        self.reg_losses_theta = tf.add_n(
            tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=ENERGY_NET_SCOPE)
        )
        # Entropy Regularization, Section 5 of Tu & Gimpel 2018
        # We add the epsilon constant for numerical stability
        epsilon = 1e-7
        prob = tf.clip_by_value(self.inference_net.layer2_out, epsilon, 1 - epsilon)
        not_prob = 1 - prob
        self.reg_losses_entropy = tf.reduce_sum(
            -1 * prob * tf.log(prob) - not_prob * tf.log(not_prob)
        )

    def create_losses(self):
        config = self.config
        batch_size = config.train.batch_size

        abs_difference = tf.reduce_sum(
            tf.square(self.labels_y - self.inference_net.layer2_out), axis=1
        )
        max_difference = tf.maximum(
            abs_difference - self.energy_net2.energy_out + self.energy_net1.energy_out,
            tf.constant([0] * batch_size, dtype=tf.float32)
        )
        self.base_objective = tf.reduce_sum(max_difference)

        lamb_reg_phi = config.train.lamb_reg_phi
        lamb_reg_theta = config.train.lamb_reg_theta
        lamb_reg_entropy = config.train.lamb_reg_entropy
        lamb_pretrain_bias = config.train.lamb_pretrain_bias

        self.cost_theta = \
            self.base_objective + \
            lamb_reg_theta * self.reg_losses_theta

        # Negative sign for reg_losses_phi since we want to maximize phi objective
        self.gain_phi = \
            self.base_objective - \
            lamb_reg_phi * self.reg_losses_phi - \
            lamb_reg_entropy * self.reg_losses_entropy - \
            lamb_pretrain_bias * self.pre_train_bias

        phi_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=INFNET_SCOPE)
        theta_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=ENERGY_NET_SCOPE)

        self.phi_opt = tf.train.AdamOptimizer(config.train.lr_phi).minimize(
            -1 * self.gain_phi, var_list=phi_vars
        )
        self.theta_opt = tf.train.AdamOptimizer(config.train.lr_theta).minimize(
            self.cost_theta, var_list=theta_vars
        )

    def infnet_losses(self):
        infnet_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=INFNET_SCOPE)

        # Minimizing a cross entropy loss for independent classes
        epsilon = 1e-7
        prob = tf.clip_by_value(self.inference_net.layer2_out, epsilon, 1 - epsilon)
        not_prob = 1 - prob
        self.infnet_ce_loss = tf.reduce_sum(
            -1 * (1 - self.labels_y) * tf.log(not_prob) - self.labels_y * tf.log(prob)
        )
        self.infnet_ce_opt = tf.train.AdamOptimizer(self.config.train.lr_psi).minimize(
            self.infnet_ce_loss, var_list=infnet_vars
        )

        # Psi parameter optimization, Equation (5) in Tu & Gimpel, 2018
        self.energy_psi = tf.reduce_sum(self.energy_net2.energy_out)
        self.psi_opt = tf.train.AdamOptimizer(self.config.train.lr_psi).minimize(
            self.energy_psi, var_list=infnet_vars
        )

    def evaluate(self):
        self.probabilities = self.inference_net.layer2_out
        self.outputs = tf.round(self.probabilities)
        self.diff = tf.cast(
            tf.reduce_sum(tf.abs(self.outputs - self.labels_y), axis=1),
            dtype=tf.int64
        )
        self.results = self.config.train.batch_size - tf.count_nonzero(self.diff)

    def init_saver(self):
        # here you initalize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
