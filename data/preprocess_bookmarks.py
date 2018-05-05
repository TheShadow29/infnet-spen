import _pickle as cPickle
import numpy as np
import sys

INPUTS = 2150
TAGS = 208

for d, v in [('train', 5), ('dev', 2), ('test', 3)]:
    dataset = []
    for i in range(1, v + 1):
        print(i)
        inputs = np.load('data/bookmarks/%s%d.npy' % (d, i))[:, :-1]
        labels = np.load('data/bookmarks/%s%d_label.npy' % (d, i))
        print(inputs.shape)
        if inputs.shape[1] != INPUTS or labels.shape[1] != TAGS:
            print("error")
            sys.exit(0)
        for j in range(len(inputs)):
            dataset.append({
                'feats': inputs[j],
                'types': labels[j]
            })
    with open('data/bookmarks/%s.pickle' % d, 'wb') as f:
        cPickle.dump(dataset, f)
