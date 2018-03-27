import numpy as np
import os
import _pickle as cPickle


def load_embeddings(config):
    data_dir = config['data_dir']
    return np.load(os.path.join(data_dir, 'reduced_embeddings.npy'))


def load_vocab(config):
    # ## TODO - Could move to this to classmethod
    data_dir = config['data_dir']

    # Loading the type vocabulary
    types = []
    with open(os.path.join(data_dir, 'types'), 'r') as f:
        for line in f:
            types.append(line.split('\t')[0].strip())
    types_vocab = {t: i for i, t in enumerate(types)}

    # Loading the entity vocabulary
    entities = []
    with open(os.path.join(data_dir, 'entities'), 'r') as f:
        for line in f:
            entities.append(line.strip())
    entities_vocab = {t: i for i, t in enumerate(entities)}

    # returning pointers to vocabularies for external usage
    return types, types_vocab, entities, entities_vocab


class FigmentDataGenerator(object):
    def __init__(self, config, split):
        self.config = config
        self.split = split
        # Load vocab explicitly when needed
        self.load_data()
        # Reset batch pointer to zero
        self.batch_pointer = 0

    def load_data(self):
        data_dir = self.config['data_dir']
        # Load the current split
        with open(os.path.join(data_dir, '%s.pickle' % self.split), 'rb') as f:
            self.data = cPickle.load(f)
        self.data_x = np.array(instance['entity_id'] for instance in self.data)
        self.data_y = np.array(instance['type_vector'] for instance in self.data)
        self.len = len(self.data)

    def next_batch(self, batch_size):
        bp = self.batch_pointer
        # This will return a smaller size if not sufficient
        # The user must pad the batch in an external API
        # Or write a TF module with variable batch size
        batch_x = self.data_x[bp:bp + batch_size]
        batch_y = self.data_y[bp:bp + batch_size]

        self.batch_pointer += batch_size
        if self.batch_pointer >= len(batch_x):
            self.batch_pointer = 0

        yield batch_x, batch_y
