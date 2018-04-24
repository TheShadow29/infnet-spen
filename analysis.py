import itertools
import re
import sys


result = re.compile(r'Accuracy\sof\sinference\snetwork\sis\s\d+\s/\s\d+\s=\s(\d*\.?\d*)')
FILENAME_TEMPLATE = "logs/phi_{0}_theta_{1}_ent_{2}_bias_{3}.log"


def add_results(instance):
    matches = re.findall(result, instance['text'])
    # Assumption of 2 evaluations per stage
    if len(matches) != 18:
        print("Error in %s" % instance['filename'])
        sys.exit(0)
    instance['results'] = [float(x) for x in matches]
    return

ranges = [0.01, 0.1, 1, 10, 100]
# Tune each of the four parameters on these values
lists = [ranges] * 4
lists = list(itertools.product(*lists))
data = []

for i, config in enumerate(lists):
    filename = FILENAME_TEMPLATE.format(config[0], config[1], config[2], config[3])
    with open(filename, 'r') as f:
        data.append({
            'text': f.read(),
            'config': config,
            'filename': filename
        })
    add_results(data[-1])

for d in data:
    if d['results'][-1] != 0.0 or d['results'][-4] != 0.0:
        print("%s, %.4f, %.4f" % (d['filename'], d['results'][-4], d['results'][-1]))
