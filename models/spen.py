import tensorflow as tf

from base.base_model import BaseModel
from models.energy_net import EnergyNet
from models.inference_net import InferenceNet


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

        self.is_training = tf.placeholder(tf.bool)

        self.input_x = tf.placeholder(
            tf.int64, shape=[batch_size]
        )
        self.labels_y = tf.placeholder(
            tf.float32, shape=[batch_size, config.type_vocab_size]
        )

        self.create_embeddings_graph()
        self.feature_input = tf.nn.embedding_lookup(self.embeddings, self.input_x)
        self.build_subnets()
        self.regularize()

        abs_difference = tf.reduce_sum(
            tf.abs(self.labels_y - self.inference_net.layer2_out), axis=1
        )
        max_difference = tf.maximum(
            abs_difference - self.energy_net2.energy_out + self.energy_net1.energy_out,
            tf.constant([0] * batch_size, dtype=tf.float32)
        )
        self.base_objective = tf.reduce_sum(max_difference)

        lamb_reg_phi = config.train.lamb_reg_phi
        lamb_reg_theta = config.train.lamb_reg_theta
        lamb_reg_entropy = config.train.lamb_reg_entropy

        self.cost_theta = \
            self.base_objective + \
            lamb_reg_theta * self.reg_losses_theta

        # Negative sign for reg_losses_phi since we want to maximize phi objective
        self.gain_phi = \
            self.base_objective - \
            lamb_reg_phi * self.reg_losses_phi - \
            lamb_reg_entropy * self.reg_losses_entropy

        phi_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="inference_net")
        theta_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="energy_net")

        self.phi_opt = tf.train.AdamOptimizer(config.train.lr_phi).minimize(
            -1 * self.gain_phi, var_list=phi_vars
        )
        self.theta_opt = tf.train.AdamOptimizer(config.train.lr_theta).minimize(
            self.cost_theta, var_list=theta_vars
        )

        self.test_cost = tf.reduce_sum(self.energy_net3.energy_out)
        shi_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="test_inference_net")
        self.shi_opt = tf.train.AdamOptimizer(config.train.lr_shi).minimize(
            self.test_cost, var_list=shi_vars
        )

        self.acc = tf.metrics.accuracy(
            labels=tf.argmax(self.labels_y, 1),
            predictions=tf.argmax(self.test_inference_net.layer2_out, 1)
        )

        return

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

        # Used in the static / non-static configurations
        self.load_embeddings = self.embeddings.assign(self.embeddings_placeholder)

    def build_subnets(self):
        config = self.config
        regularizer = tf.contrib.layers.l2_regularizer(1.0)

        # This is the cost-augmented inference network
        # This is used for GAN-style training
        with tf.variable_scope("inference_net", regularizer=regularizer):
            self.inference_net = InferenceNet(
                config, self.feature_input
            )
        # This is the actual inference network
        # Corresponds to psi parameters in Tu & Gimpel 2018
        with tf.variable_scope("test_inference_net", regularizer=regularizer):
            self.test_inference_net = InferenceNet(
                config, self.feature_input
            )

        with tf.variable_scope("energy_net", regularizer=regularizer):
            self.energy_net1 = EnergyNet(
                config, self.feature_input, self.labels_y
            )
        with tf.variable_scope("energy_net", reuse=True, regularizer=regularizer):
            self.energy_net2 = EnergyNet(
                config, self.feature_input, self.inference_net.layer2_out
            )
        with tf.variable_scope("energy_net", reuse=True, regularizer=regularizer):
            self.energy_net3 = EnergyNet(
                config, self.feature_input, self.test_inference_net.layer2_out
            )

    def regularize(self):
        self.reg_losses_phi = tf.add_n(
            tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope="inference_net")
        )
        self.reg_losses_theta = tf.add_n(
            tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope="energy_net")
        )
        # Entropy Regularization, Section 5 of Tu & Gimpel 2018
        prob = self.inference_net.layer2_out
        not_prob = 1 - self.inference_net.layer2_out
        self.reg_losses_entropy = tf.reduce_sum(
            -1 * prob * tf.log(prob) - not_prob * tf.log(not_prob)
        )

    def init_saver(self):
        # here you initalize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
