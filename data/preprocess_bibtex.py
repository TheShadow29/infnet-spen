import _pickle as cPickle
import numpy as np

INPUTS = 1836
TAGS = 159

# Loading all the datasets data
dataset_names = ['train', 'test']
datasets = {name: [] for name in dataset_names}
for d in dataset_names:
    with open('data/bibtex/bibtex-%s.arff' % d, 'r') as f:
        data_started = False
        for line in f:
            if line.strip() == "@data":
                data_started = True
                continue
            if data_started is False:
                continue
            line = line.replace("{", "")
            line = line.replace("}", "")
            tokens = [int(x.split()[0]) for x in line.split(",")]
            datasets[d].append({
                'sparse_feats': [t for t in tokens if t < INPUTS],
                'sparse_types': [t - INPUTS for t in tokens if t >= INPUTS]
            })

# Copying test set to form dev set
datasets['dev'] = datasets['test']

for name, dataset in datasets.items():
    for instance in dataset:
        feats_vector = np.zeros(INPUTS)
        feats_vector[instance['sparse_feats']] = 1
        types_vector = np.zeros(TAGS)
        types_vector[instance['sparse_types']] = 1
        instance['feats'] = feats_vector
        instance['types'] = types_vector
    with open('data/bibtex/%s.pickle' % name, 'wb') as f:
        cPickle.dump(dataset, f)
