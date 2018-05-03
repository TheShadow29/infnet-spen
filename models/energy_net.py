import tensorflow as tf


class EnergyNet(object):
    def __init__(self, config, input_x, input_y):
        """ This models the energy function in the SPEN framework.
        Takes as an input x and y and produces an energy value.
        Scope is necessary while deciding whether or not to re-use variables

        input_x: set of feature vectors
        input_y: one hot style type classes
        """
        self.input_x = input_x
        self.input_y = input_y
        self.config = config
        self.type_vocab_size = config.type_vocab_size
        self.batch_size, self.feature_size = input_x.get_shape()

        self.build_model()

    def build_model(self):
        label_measurements = self.config.label_measurements

        # creating the linear energy model
        with tf.name_scope('global_energy'):
            self.linear_wt = tf.get_variable(
                "linear_wt", [self.feature_size, self.type_vocab_size]
            )
            self.negative_logits = tf.matmul(self.input_x, self.linear_wt)
            # -1 since we wish to maximize probability, but b_i designed to minimize energy
            self.pretrain_probs = tf.sigmoid(-1 * self.negative_logits)
            # Equation 1, Tu & Gimpel 2018
            self.linear_out = tf.reduce_sum(
                tf.multiply(self.negative_logits, self.input_y),
                axis=1
            )

        # creating the label energy function
        with tf.name_scope('label_energy'):
            # Equation 2, Tu & Gimpel 2018
            self.label_energy1 = tf.layers.dense(
                inputs=self.input_y,
                units=label_measurements,
                activation=tf.nn.softplus,
                name='label_energy1'
            )
            self.label_energy2 = tf.squeeze(tf.layers.dense(
                inputs=self.label_energy1,
                units=1,
                activation=None,
                name='label_energy2',
                use_bias=False
            ))

        self.energy_out = self.linear_out + self.label_energy2
