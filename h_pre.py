import json
import os
import sys

hs_folder = sys.argv[2] if len(sys.argv) > 2 else "hearthstone"
train_file_name = "train_hs"
test_file_name = "test_hs"
dev_file_name = "dev_hs"

def read_samples(file_name):
    with open(os.path.join(hs_folder, file_name + ".in"), 'r') as f:
        train_source_lines = f.read().splitlines()

    with open(os.path.join(hs_folder, file_name + ".out"), 'r') as f:
        train_target_lines = f.read().splitlines()    

    return [{"source": s, "target": t.replace("§", "\n").replace("\\ ", "")} for (s, t) in zip(train_source_lines, train_target_lines)]

train_set = read_samples(train_file_name)
dev_set = read_samples(dev_file_name)
test_set = read_samples(test_file_name)

def dump_jsonl(ds, file):
    with open(file, 'w') as f:
        f.write('\n'.join([json.dumps(l) for l in ds]))

dump_jsonl(train_set, 'hearthstone/train.jsonl')
dump_jsonl(test_set, 'hearthstone/test.jsonl')
dump_jsonl(dev_set, 'hearthstone/dev.jsonl')
