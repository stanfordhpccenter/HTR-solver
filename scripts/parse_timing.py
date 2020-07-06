#!/usr/bin/env python2

import argparse
import os
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("infile", type=argparse.FileType("r"))
args = parser.parse_args()

task = []
Lines = []
time = []

# Parse input file
line = args.infile.readline()
while line:
   words = line.split(" ")
   if (words[0] == "TIMING:"):
      task.append(words[1])
      Lines.append(words[6])
      time.append(float(words[9]))
   line = args.infile.readline()

assert len(task) == len(time)
assert len(Lines) == len(time)
print("{} timing lines parsed".format(len(Lines)))

# Sort time intervals by tasks and lines
timing = dict([])
prevVals = dict([])
prevTime = time[0]
prevLine = Lines[0]
prevTime = time[0]

for i in range(len(Lines)):
   if task[i] in timing:
      # We have already encountered this task
      assert task[i] in prevVals

      key = prevVals[task[i]][0] + "-" + Lines[i]
      deltaT = time[i] - prevVals[task[i]][1]

      if (deltaT < 0.0):
         print(task[i], prevVals[task[i]][0], Lines[i])

      if key in timing[task[i]]:
         # We already have this entry in the record
         timing[task[i]][key][0] += deltaT
         timing[task[i]][key][1] += 1

      else :
         # Add the entry to the record
         timing[task[i]][key] = [deltaT, 1]

      # Store previous values of lines and time
      prevVals[task[i]][0] = Lines[i]
      prevVals[task[i]][1] = time[i]

   else:
      # Start storing values for this task
      timing[task[i]] = dict([])
      prevVals[task[i]] = [Lines[i], time[i]]

# Compute average deltaT
for task in timing:
   for key in timing[task]:
      timing[task][key] = timing[task][key][0]/timing[task][key][1]

# Plot hystograms
for i, task in enumerate(timing):
   plt.figure(i)
   plt.title("Task: {}".format(task))
   plt.bar(list(timing[task]), list(timing[task].values()))
   plt.xticks(rotation=70)

plt.show()
