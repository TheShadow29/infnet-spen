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
        regularizer = tf.contrib.layers.l2_regularizer(1.0)

        self.is_training = tf.placeholder(tf.bool)

        self.input_x = tf.placeholder(
            tf.int64, shape=[config.train.batch_size]
        )
        self.labels_y = tf.placeholder(
            tf.float32, shape=[config.train.batch_size, config.type_vocab_size]
        )

        self.create_embeddings_graph()
        self.feature_input = tf.nn.embedding_lookup(self.embeddings, self.input_x)

        with tf.variable_scope("energy_net", regularizer=regularizer):
            self.energy_net1 = EnergyNet(
                config, self.feature_input, self.labels_y
            )

        with tf.variable_scope("inference_net", regularizer=regularizer):
            self.inference_net = InferenceNet(
                config, self.feature_input
            )

        with tf.variable_scope("energy_net", reuse=True, regularizer=regularizer):
            self.energy_net2 = EnergyNet(
                config, self.feature_input, self.inference_net.layer2_out
            )

        # Loss functions for Eqn7
        # Eqn8, replacing structured hinge loss with absolute_difference
        # To maximize
        lamb_reg_phi = config.lamb_reg_phi
        lamb_reg_theta = config.lamb_reg_theta
        reg_losses_phi = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                           scope="inference_net")
        reg_losses_theta = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                             scope="energy_net")
        self.cost = (tf.maximum(tf.losses.absolute_difference(self.labels_y, self.inference_out)
                                - self.energy_infer + self.energy_truth, tf.constant([0]))
                     + lamb_reg_theta * tf.add_n(reg_losses_theta)
                     - lamb_reg_phi * tf.add_n(reg_losses_phi))

        phi_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="inference_net")
        theta_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="energy_net")

        self.phi_opt = tf.train.AdamOptimizer(
            config.train.lr).minimize(
                -self.cost, var_list=phi_vars)
        self.theta_opt = tf.train.AdamOptimizer(
            config.train.lr).minimize(self.cost, var_list=theta_vars)

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

    def init_saver(self):
        # here you initalize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
