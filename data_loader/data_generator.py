import numpy as np
from pathlib import Path
from collections import defaultdict
import pdb


class data_generator:
    def __init__(self, config):
        self.config = config
        self.decode_config()
        # load data here
        # self.input = np.ones((500, 784))
        # self.y = np.ones((500, 10))

    def decode_config(self):
        raise NotImplementedError

    def next_batch(self, batch_size):
        raise NotImplementedError


class figment_data_generator(data_generator):
    def decode_config(self):
        # config has the following:
        # tdir (in Path format), types(str), Efile, embfile
        # ds (train/test/dev)
        self.tdir = Path(self.config['tdir'])
        self.type_file = self.tdir / self.config['type_file']
        self.ent_file = self.tdir / self.config['entity_file']
        self.emb_file = self.tdir / self.config['embedding_file']
        self.ds = self.config['ds']

        self.target_to_ind()
        self.get_word_vectors()
        self.fill_entity_data()

        return

    def next_batch(self, batch_size):
        idx = np.random.choice(500, batch_size)
        # pdb.set_trace()
        yield self.data_mat[idx, :], self.all_types_result_vector[idx]

    def target_to_ind(self):
        target_ind_map = dict()
        type_freq_train_dev = dict()
        c = 0
        with self.type_file.open(mode='r') as f:
            for line in f:
                t = line.split('\t')[0].strip()
                target_ind_map[t] = c
                if len(line.split('\t')) < 3:
                    type_freq = 0
                else:
                    type_freq = line.split('\t')[2].strip()
                if type_freq == 'null':
                    type_freq_train_dev[t] = 0
                else:
                    type_freq_train_dev[t] = int(type_freq)
                c += 1
        # return (target_ind_map, len(target_ind_map), type_freq_train_dev)
        self.type2ind = target_ind_map
        self.num_targets = len(target_ind_map)
        self.type_freq_train_dev = type_freq_train_dev
        return

    def get_word_vectors(self):
        print('loading word vectors')
        with self.emb_file.open(mode='r') as f:
            print('now open ', self.emb_file)
            emb_str = f.read()

        emb_lines = emb_str.split('\n')
        if len(emb_lines[-1].strip()) == 0:
            emb_lines = emb_lines[:-1]

        word_vecs = defaultdict(list)
        # -1 because need to remove first word
        vec_size = len(emb_lines[0].split()) - 1
        word_vecs['<UNK>'] = [0.001 for _ in range(vec_size)]

        # ####Need to confirm if L195 in figment/myutils is required
        for el in emb_lines:
            emb_line = el.strip()
            emb_line_parts = emb_line.split()
            word = emb_line_parts[0].strip()
            emb_line_parts.pop(0)
            word_vecs[word] = list(map(float, emb_line_parts))

        print('vector size is ', str(vec_size))
        print('len(word_vecs) ', len(word_vecs))

        # return (word_vecs, vec_size)
        # pdb.set_trace()
        self.word_vecs = word_vecs
        self.vec_size = vec_size
        return

    def fill_entity_data(self, bin_out_vec_bool=True):
        with self.ent_file.open(mode='r') as f:
            print('now open ', self.ent_file)
            ent_str = f.read()
        ent_lines = ent_str.split('\n')
        if len(ent_lines[-1].strip()) == 0:
            ent_lines = ent_lines[:-1]

        input_entities = []
        result_vector = []
        all_types_result_vector = []

        num_inps = len(ent_lines)
        data_mat = np.zeros(shape=(num_inps, self.vec_size))

        # c = 0
        i1 = 0
        i2 = 0
        for c, ent_l in enumerate(ent_lines):
            ent_line = ent_l.strip()
            ent_line_parts = ent_line.split('\t')
            ent = ent_line_parts[0]
            target = ent_line_parts[1]
            if target not in self.type2ind and self.ds != 'test':
                print('type not found', target)
                continue
            types_ind = []
            if len(ent_line_parts) >= 3:
                other_types = ent_line_parts[2].split(' ')
                for i in range(0, len(other_types)):
                    if other_types[i] not in self.type2ind:
                        continue
                    types_ind.append(self.type2ind[other_types[i]])
            # ##Need to check if L332 in figment/myutils is required
            # pdb.set_trace()
            if ent not in self.word_vecs:
                # pdb.set_trace()
                print(ent, 'not in word_vecs')
                i1 += 1
                # if self.ds == 'test':
                # data_mat[c, :] =
            else:
                # pdb.set_trace()
                data_mat[c, :] = self.word_vecs[ent]
            i2 += 1
            input_entities.append(ent)

            if target in self.type2ind:
                result_vector.append(self.type2ind[target])
                types_ind.append(self.type2ind[target])
            else:
                result_vector.append(0)

            bin_vec = self.convert_target_to_bin(types_ind, self.num_targets)
            if bin_out_vec_bool:
                all_types_result_vector.append(bin_vec)
            else:
                all_types_result_vector.append(types_ind)

        self.result_vector = np.array(result_vector)
        self.data_mat = data_mat
        self.input_entities = np.array(input_entities)
        self.all_types_result_vector = np.array(all_types_result_vector)
        # return result_vector, data_mat, input_entities, all_types_result_vector
        # pdb.set_trace()
        return

    def convert_target_to_bin(self, other_types_ind, n_out):
        outvec = np.zeros(n_out, np.int32)
        #     outvec[nt_ind] = 1
        for ind in other_types_ind:
            outvec[ind] = 1
        return outvec
