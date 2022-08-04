#!/usr/bin/env python3

import argparse
import collections
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--num_times", type=int, default=4)
parser.add_argument('-b', '--before', type=int, default=5)
parser.add_argument("-out", "--baseDir")
args = parser.parse_args()

def postFile(Dir):
   lineno = 0
   t_start = None
   tail = collections.deque()

   with open(Dir + "/sample0/console.txt", "r") as fout:
      for line in fout:
         lineno += 1
         if lineno - 2 < args.before:
            continue
         t = float(line.split()[2])
         if t_start is None:
            t_start = t
         tail.append(t)
   assert t_start is not None
   t_end = tail[-1]
   return t_end - t_start

nodes = []
wt = []
for i in range(0,args.num_times):
   nodes.append(int(2**i))
   wt.append(postFile(args.baseDir + "/" + str(nodes[i])))

scale = wt[0]
optimal = []
efficiency = []
for i in range(0,args.num_times):
   optimal.append(1)
   efficiency.append(scale/wt[i])
   print(nodes[i], wt[i], efficiency[i])

plt.figure(1)
plt.semilogx(nodes, optimal,    '--k', label="Optimal")
plt.semilogx(nodes, efficiency, '-ok', label="Measured")
plt.xlabel(r"nodes")
plt.ylabel(r"efficiency")
plt.xticks(nodes)
plt.minorticks_off()
plt.grid()
plt.legend()
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%d'))
plt.gca().set_ylim(0.6, 1.05)
plt.savefig(args.baseDir+".eps", bbox_inches='tight')

plt.show()

