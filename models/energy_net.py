import tensorflow as tf


class EnergyNet(object):
    def __init__(self, config, input_x, input_y):
        """ This models the energy function in the SPEN framework.
        Takes as an input x and y and produces an energy value.
        Scope is necessary while deciding whether or not to re-use variables

        input_x: set of embedding vectors
        input_y: one hot style type classes
        """
        self.input_x = input_x
        self.input_y = input_y
        self.config = config
        self.type_vocab_len = config.type_vocab_size
        # get_shape giving too many values to unpack
        # self.batch_size, self.embedding_size = input_x.get_shape()
        print('Input Dim', input_x.get_shape())
        self.batch_size, self.embedding_size = input_x.get_shape()

        self.build_model()

    def build_model(self):
        # creating the linear energy model
        # with tf.name_scope("dummy"):
        self.linear_wt = tf.get_variable("linear_wt", [self.embedding_size, self.type_vocab_len])
        # Equation 1, Tu & Gimpel 2018
        self.linear_out = tf.reduce_sum(
            tf.multiply(tf.matmul(self.input_x, self.linear_wt), self.input_y),
            axis=1
        )

        # creating the label energy function
        # C1 in eqn2
        self.label_matrix = label_matrix = tf.get_variable(
            "label_matrix", [self.type_vocab_len, self.type_vocab_len]
        )
        # c2 in eqn2
        self.label_vector = label_vector = tf.get_variable(
            "label_vector", [1, self.type_vocab_len]
        )

        # Equation 2, Tu & Gimpel 2018
        # Hard-coding tanh non-linearity for now
        label_temp1 = tf.matmul(self.input_y, label_matrix)
        self.label_out = tf.reduce_sum(
            tf.multiply(tf.tanh(label_temp1), label_vector),
            axis=1
        )

        self.energy_out = self.linear_out + self.label_out
