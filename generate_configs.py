import itertools

BASE_CONFIG = \
    """
{{
    "exp_name": "figment_sq_diff",
    "data":{{
        "dataset": "figment",
        "data_dir": "data/figment/",
        "splits": ["Etrain", "Edev", "Etest"],
        "embeddings": true,
        "vocab": true,
        "data_generator": "FigmentDataGenerator"
    }},
    "tensorboard": true,
    "feature_size": 200,
    "label_measurements": 15,
    "type_vocab_size": 102,
    "entities_vocab_size": 201933,
    "embeddings_tune": false,
    "max_to_keep": 5,
    "num_epochs": [2, 2, 5],
    "train": {{
        "diff_type": "sq_diff",
        "batch_size": 32,
        "state_size": [784],
        "max_to_keep": 5,
        "hidden_units": 200,
        "lr_phi": 0.001,
        "lr_theta": 0.001,
        "lr_psi": 0.001,
        "lamb_reg_phi": {0},
        "lamb_reg_theta": {1},
        "lamb_reg_entropy": {2},
        "lamb_pretrain_bias": {3}
    }}
}}"""

FILENAME_TEMPLATE = "phi_{0}_theta_{1}_ent_{2}_bias_{3}"

ranges = [0.01, 0.1, 1, 10, 100]
# Tune each of the four parameters on these values
lists = [ranges] * 4
lists = list(itertools.product(*lists))

for i, config in enumerate(lists):
    config_text = BASE_CONFIG.format(config[0], config[1], config[2], config[3])
    filename = FILENAME_TEMPLATE.format(config[0], config[1], config[2], config[3])
    with open('configs/%s.json' % filename, 'w') as f:
        f.write(config_text)
