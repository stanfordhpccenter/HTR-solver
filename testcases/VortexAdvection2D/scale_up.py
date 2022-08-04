#!/usr/bin/env python3

import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('base_json', type=argparse.FileType('r'), default='base.json')
parser.add_argument('-n', '--num_times', type=int, default=4)
args = parser.parse_args()

# Read base config
config = json.load(args.base_json)
assert int(config['Mapping']['tiles'][0]) / int(config['Mapping']['tilesPerRank'][0]) == 1
assert int(config['Mapping']['tiles'][1]) / int(config['Mapping']['tilesPerRank'][1]) == 1
assert int(config['Mapping']['tiles'][2]) / int(config['Mapping']['tilesPerRank'][2]) == 1

# Scale up
with open('VortexAdvection2D_' + str(config['Grid']['xNum']) + '.json', 'w') as fout:
    json.dump(config, fout, indent=3)

for i in range(1,args.num_times):
    config['Grid']['xNum'] *= 2
    config['Grid']['yNum'] *= 2
    config['Integrator']['maxIter'] *=2
    config['Integrator']['TimeStep']['DeltaTime'] /=2
    with open('VortexAdvection2D_' + str(config['Grid']['xNum']) + '.json', 'w') as fout:
        json.dump(config, fout, indent=3)
