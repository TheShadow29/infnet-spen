import tensorflow as tf


class InferenceNet(object):
    def __init__(self, config, input_x):
        """ This models calcuates the inferred y value for a given x.
        It's a simple two layer feedforward neural network, ending with sigmoid
        """
        self.input_x = input_x
        self.config = config
        self.batch_size, self.embedding_size = tf.get_shape(input_x)
        self.type_vocab_len = config['type_vocab_len']
        self.build_model()

    def build_model(self):
        self.layer1_out = tf.layers.dense(
            input=self.input_x,
            units=self.config['hidden_units'],
            activation=tf.nn.relu
        )

        self.layer2_out = tf.layers.dense(
            input=self.layer1_out,
            units=self.type_vocab_len,
            activation=tf.sigmoid
        )
