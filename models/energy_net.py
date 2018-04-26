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
            # Equation 1, Tu & Gimpel 2018
            self.linear_out = tf.reduce_sum(
                tf.multiply(tf.matmul(self.input_x, self.linear_wt), self.input_y),
                axis=1
            )

        # creating the label energy function
        with tf.name_scope('label_energy'):
            # C1 in eqn2
            self.label_matrix = label_matrix = tf.get_variable(
                "label_matrix", [self.type_vocab_size, label_measurements]
            )
            # c2 in eqn2
            self.label_vector = label_vector = tf.get_variable(
                "label_vector", [label_measurements]
            )

            # Equation 2, Tu & Gimpel 2018
            # Hard-coding softplus non-linearity for now
            label_temp1 = tf.matmul(self.input_y, label_matrix)
            self.label_out = tf.reduce_sum(
                tf.multiply(tf.nn.softplus(label_temp1), label_vector),
                axis=1
            )

        self.energy_out = self.linear_out + self.label_out
