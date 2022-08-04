#!/usr/bin/env python3

import argparse
import json
import os
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("base_json", type=argparse.FileType("r"), default="base.json")
parser.add_argument("-n", "--num_times", type=int, default=4)
parser.add_argument("-out", "--baseDir")
args = parser.parse_args()

# Read base config
config = json.load(args.base_json)
assert int(config["Mapping"]["tiles"][0]) / int(config["Mapping"]["tilesPerRank"][0]) == 1
assert int(config["Mapping"]["tiles"][1]) / int(config["Mapping"]["tilesPerRank"][1]) == 1
assert int(config["Mapping"]["tiles"][2]) / int(config["Mapping"]["tilesPerRank"][2]) == 1

baseDir = os.path.expandvars("$PWD") + "/" + args.baseDir
if not os.path.exists(baseDir): os.makedirs(baseDir)

# Scale up
for i in range(0,args.num_times):
   nodes = int(config["Mapping"]["tiles"][1]/config["Mapping"]["tilesPerRank"][1])

   with open(baseDir + "/" + str(nodes) + ".json", "w") as fout:
      json.dump(config, fout, indent=3)

   outDir = baseDir + "/" + str(nodes)
   if not os.path.exists(outDir): os.makedirs(outDir)

   command = "{} -i {} -o {}".format(os.path.expandvars("PROFILE=1 $HTR_DIR/prometeo.sh"),
                                     baseDir + "/" + str(nodes) + ".json", outDir)
   print(command)
   subprocess.call(command, shell=True)

   config["Grid"]["yNum"] *= 2
   config["Grid"]["GridInput"]["width"][1] *= 2
   config["Mapping"]["tiles"][1] *=2

