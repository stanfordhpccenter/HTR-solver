#!/usr/bin/env python3

import argparse
import json
import os
import subprocess, sys
import copy

parser = argparse.ArgumentParser()
parser.add_argument("base_json", type=argparse.FileType("r"), default="base.json")
parser.add_argument("--nmin", type=int, default=0)
parser.add_argument("-n", "--num_times", type=int, default=4)
parser.add_argument("-out", "--baseDir")
parser.add_argument("--lp", nargs='+', help='Low priority samples')
parser.add_argument("--debug", action='store_true')
parser.add_argument("--seandeb", action='store_true')
parser.add_argument("--profile", action='store_true')
args = parser.parse_args()

# Read base config
config = json.load(args.base_json)
assert int(config["Mapping"]["tiles"][0]) / int(config["Mapping"]["tilesPerRank"][0]) == 1
assert int(config["Mapping"]["tiles"][1]) / int(config["Mapping"]["tilesPerRank"][1]) == 1
assert int(config["Mapping"]["tiles"][2]) / int(config["Mapping"]["tilesPerRank"][2]) == 1

baseDir = os.path.expandvars("$PWD") + "/" + args.baseDir
if not os.path.exists(baseDir): os.makedirs(baseDir)

configOrig = copy.deepcopy(config)

# Scale up
for i in range(args.nmin, args.num_times):
   config = copy.deepcopy(configOrig)
   config["Grid"]["yNum"] *= pow(2, i)
   config["Grid"]["GridInput"]["width"][1] *= pow(2, i)
   config["Mapping"]["tiles"][1] *= pow(2, i)

   #nodes = int(config["Mapping"]["tiles"][0]/config["Mapping"]["tilesPerRank"][0])
   nodes = int(config["Mapping"]["tiles"][1]/config["Mapping"]["tilesPerRank"][1])

   with open(baseDir + "/" + str(nodes) + ".json", "w") as fout:
      json.dump(config, fout, indent=3)

   outDir = baseDir + "/" + str(nodes)
   if not os.path.exists(outDir): os.makedirs(outDir)

   #command = "{} -i {} -o {}".format(os.path.expandvars("PROFILE=1 $HTR_DIR/prometeo.sh"),
   command = ""
   if args.profile:
       command += "PROFILE=1 "
   #
   command += "{} -i {} -o {} ".format(os.path.expandvars("$HTR_DIR/prometeo.sh"),
                                     baseDir + "/" + str(nodes) + ".json", outDir)

   if args.lp:
      # increase lp as the number of nodes increase
      for n in range(0,nodes):
          for index, v in enumerate(args.lp):
             print("args.lp:",index, v)
             # read lf file
             with open(v, 'r') as fin:
                conf_lf = json.load(fin)
             #
             conf_lf_name = baseDir + "/lf_" + os.path.splitext(os.path.basename(v))[0] + "_" + str(index) + "_" + str(nodes) + ".json"
             with open(conf_lf_name, "w") as fout:
                json.dump(conf_lf, fout, indent=3)
             #
     
             command += " -lp "+conf_lf_name+" "
          ##
      ##
   #

   if args.debug:
       command += " -level prometeo_mapper=1 -logfile "+ outDir + "/mapper_%.log "
   #
   if args.seandeb:
       command += " -level dma=2,xplan=1 "
   #

   print(command)
   subprocess.call(command, shell=True)

   #config["Grid"]["xNum"] *= 2
   #config["Grid"]["xWidth"] *= 2
   #config["Mapping"]["tiles"][0] *=2 

   #config["Grid"]["yNum"] *= 2
   #config["Grid"]["yWidth"] *= 2
   #config["Mapping"]["tiles"][1] *=2 

