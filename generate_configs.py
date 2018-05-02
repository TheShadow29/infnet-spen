import itertools

BASE_CONFIG = \
    """
{{
    "exp_name": "bibtex_sq_diff",
    "data":{{
        "dataset": "bibtex",
        "data_dir": "data/bibtex/",
        "splits": ["train", "dev", "test"],
        "embeddings": false,
        "vocab": false,
        "finput_dim": 1836,
        "data_generator": "BibtexDataGenerator"
    }},
    "tensorboard": true,
    "feature_size": 200,
    "label_measurements": 15,
    "type_vocab_size": 159,
    "entities_vocab_size": 201933,
    "embeddings_tune": false,
    "max_to_keep": 5,
    "num_epochs": [4, 2, 2],
    "train": {{
        "diff_type": "sq_diff",
        "batch_size": 32,
        "state_size": [784],
        "max_to_keep": 5,
        "hidden_units": 150,
        "lr_feat": 0.001,
        "lr_phi": 0.001,
        "lr_theta": 0.001,
        "lr_psi": 0.001,
        "lamb_reg_feats": 0,
        "lamb_reg_phi": {0},
        "lamb_reg_theta": {1},
        "lamb_reg_entropy": {2},
        "lamb_pretrain_bias": {3}
    }},
    "ssvm": {{
        "enable": false,
        "steps": 10,
        "eval": false,
        "lr_inference": 0.1
    }}
}}"""

FILENAME_TEMPLATE = "phi_{0}_theta_{1}_ent_{2}_bias_{3}"

ranges = [0.001, 0.01, 0.1, 1]
# Tune each of the four parameters on these values
lists = [ranges] * 4
lists = list(itertools.product(*lists))

for i, config in enumerate(lists):
    config_text = BASE_CONFIG.format(config[0], config[1], config[2], config[3])
    filename = FILENAME_TEMPLATE.format(config[0], config[1], config[2], config[3])
    with open('configs/%s.json' % filename, 'w') as f:
        f.write(config_text)
