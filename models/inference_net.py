import tensorflow as tf


class InferenceNet(object):
    def __init__(self, config, input_x):
        """ This models calcuates the inferred y value for a given x.
        It's a simple two layer feedforward neural network, ending with sigmoid
        """
        self.input_x = input_x
        self.config = config
        self.batch_size, self.feature_size = input_x.get_shape()
        self.type_vocab_size = config.type_vocab_size
        self.build_model()

    def build_model(self):
        self.layer1_out = tf.layers.dense(
            inputs=self.input_x,
            units=self.config.train.hidden_units,
            activation=tf.nn.relu,
            name='layer1'
        )

        self.layer2_out = tf.layers.dense(
            inputs=self.layer1_out,
            units=self.config.train.hidden_units,
            activation=tf.nn.relu,
            name='layer2'
        )

        self.layer3_out = tf.layers.dense(
            inputs=self.layer2_out,
            units=self.type_vocab_size,
            activation=None,
            name='layer3'
        )
        self.probs = tf.sigmoid(self.layer3_out)
