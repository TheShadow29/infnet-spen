import itertools
import os

FILENAME_TEMPLATE = "phi_{0}_theta_{1}_ent_{2}_bias_{3}"
SKIP_TILL = 34

ranges = [0.01, 0.1, 1, 10, 100]
# Tune each of the four parameters on these values
lists = [ranges] * 4
lists = list(itertools.product(*lists))

for i, config in enumerate(lists):
    if i < SKIP_TILL:
        continue
    print("Experiment %d / %d" % (i + 1, len(lists)))
    filename = FILENAME_TEMPLATE.format(config[0], config[1], config[2], config[3])
    os.system("python -m mains.infnet --config configs/{0}.json 2>&1 > logs/{0}.log".format(filename))
