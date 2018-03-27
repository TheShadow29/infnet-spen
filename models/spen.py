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
        self.is_training = tf.placeholder(tf.bool)

        self.input_x = tf.placeholder(
            tf.int64, shape=[config['batch_size'], config['embedding_size']]
        )
        self.labels_y = tf.placeholder(
            tf.float32, shape=[config['batch_size'], config['type_vocab_len']]
        )

        self.create_embeddings_graph()
        self.feature_input = tf.nn.embedding_lookup(self.embeddings, self.inputs)

        with tf.variable_scope("energy_net"):
            self.energy_truth = EnergyNet(config, self.feature_input, self.labels_y)

        with tf.variable_scope("inference_net"):
            self.inference_out = InferenceNet(config, self.feature_input)

        with tf.variable_scope("energy_net", reuse=True):
            self.energy_infer = EnergyNet(config, self.feature_input, self.inference_out)

    def create_embeddings_graph(self):
        config = self.config
        vocab_size = config['entities_vocab_len']
        e_size = config['embedding_size']
        # Logic for embeddings
        self.embeddings_placeholder = tf.placeholder(
            tf.float32, [vocab_size, e_size]
        )
        self.embeddings = tf.get_variable(
            "embedding", [vocab_size, e_size],
            initializer=random_uniform(0.25),
            trainable=config['embeddings_tune']
        )

        # Used in the static / non-static configurations
        self.load_embeddings = self.embeddings.assign(self.embeddings_placeholder)

    def init_saver(self):
        # here you initalize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
