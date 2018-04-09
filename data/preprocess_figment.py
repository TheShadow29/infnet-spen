import _pickle as cPickle
import numpy as np
import os
import pdb
import json
np.random.seed(0)

EMBEDDING_SIZE = 200
EMBEDDINGS = 1169307
DONT_WRITE = True
# Preprocessing embeddings file
embeddings = np.zeros((EMBEDDINGS, EMBEDDING_SIZE))
entities = []
with open('data/figment/embeddings.txt', 'r') as f:
    for i, line in enumerate(f):
        if i % 100000 == 0:
            print("%d / %d embeddings done" % (i, EMBEDDINGS))
        tokens = line.split()
        entities.append(tokens[0])
        embeddings[i] = np.array(list(map(float, tokens[1:])))
entity_vocab = {t: i for i, t in enumerate(entities)}

# Loading the types data
types = []
with open('data/figment/types', 'r') as f:
    for line in f:
        types.append(line.split('\t')[0].strip())
types_vocab = {t: i for i, t in enumerate(types)}

# Loading all the datasets data
dataset_names = ['Etrain', 'Edev', 'Etest']
entities_needed = set()
datasets = {name: [] for name in dataset_names}
for d in dataset_names:
    if 'Etest' == d:
        with open(os.path.join('data/figment', d), 'r') as f:
            for line in f:
                tokens = line.split()
                datasets[d].append({
                    'entity': tokens[0],
                    'types': tokens[1:-1]   # removing last token since it's not a tag
                })
                entities_needed.add(tokens[0])
    else:
        with open(os.path.join('data/figment', d), 'r') as f:
            for line in f:
                tokens = line.split()
                datasets[d].append({
                    'entity': tokens[0],
                    'types': tokens[1:]   # removing last token since it's not a tag
                })
                entities_needed.add(tokens[0])

reduced_entities = list(entities_needed)

reduced_entity_vocab = {t: i for i, t in enumerate(reduced_entities)}

# Now, store only those embeddings which are really needed
reduced_embeddings = np.zeros([len(reduced_entities), EMBEDDING_SIZE])
for i, entity in enumerate(reduced_entities):
    if entity in entity_vocab:
        reduced_embeddings[i] = embeddings[entity_vocab[entity]]
    else:
        print("Storing entity %s as random vector" % entity)
        reduced_embeddings[i] = np.random.uniform(-0.14, 0.14, EMBEDDING_SIZE)

# Storing the reduced vocab and embeddings
np.save('data/figment/reduced_embeddings.npy', reduced_embeddings)
if not DONT_WRITE:
    with open('data/figment/entities', 'w') as f:
        f.write('\n'.join(reduced_entities))

# Storing preprocessed pickle versions of dataset
with open('data/figment/red_map.txt', 'w') as g:
            g.write(json.dumps(reduced_entity_vocab))

for name, dataset in datasets.items():
    for instance in dataset:
        # pdb.set_trace()
        instance['entity_id'] = reduced_entity_vocab[instance['entity']]
        type_ids = [types_vocab[x] for x in instance['types']]
        instance['type_vector'] = np.zeros(len(types))
        instance['type_vector'][type_ids] = 1
    if not DONT_WRITE:
        with open('data/figment/%s.pickle' % name, 'wb') as f:
            cPickle.dump(dataset, f)
    else:
        print('DONT_WRITE is set to True')
