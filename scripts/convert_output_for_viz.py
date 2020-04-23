import os
import shutil
import sys
import subprocess
import argparse
import glob

parser = argparse.ArgumentParser()
#parser.add_argument('-s', '--section', choices=['1','2'],
#                    help='which section to visualize (if multi-section sim)')
parser.add_argument('json_file',
                    help='original simulation configuration file')
parser.add_argument('--sampledir', nargs='?', const='.', default='.',
                    help='directory with all the simulation output')
args = parser.parse_args()

sample_dir = args.sampledir
merge_dir  = os.path.join(sample_dir,'merged_data')
out_dir    = os.path.join(sample_dir,'viz_ready_data')

print('##############################################################################')
print('                            Merge parallel HDF files')
print('##############################################################################')

if not os.path.exists(merge_dir):
    os.makedirs(merge_dir)

snapshots = glob.glob(os.path.join(sample_dir,"fluid_iter*"))

for i, sn in enumerate(snapshots):
   filename = sn.split('/')[-1]+".hdf"

   if not(os.path.isfile(os.path.join(merge_dir,filename))):
      tiles = glob.glob(os.path.join(sn,"*hdf"))
      merge_fluid_command = 'python {} {}'.format(
            os.path.expandvars('$HTR_DIR/scripts/makeVirtualLayout.py'), 
            " ".join(str(x) for x in tiles))
      try:
         subprocess.call(merge_fluid_command, shell=True)
      except OSError:
         print("Failed command: {}".format(merge_fluid_command))
      else:
         print("Successfully ran command: {}".format(merge_fluid_command))

      # Move the generated files to the output directory
      try:  
         subprocess.call('mv *.hdf {}'.format(os.path.join(merge_dir, filename)), shell=True)
      except OSError:  
         print("Failed to move hdf file to: {}".format(merge_dir))
         sys.exit()
      else:  
         print("Successfully moved hdf file to: {}".format(merge_dir))

print('##############################################################################')
print('                          Generate fluid viz files ')
print('##############################################################################')

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

viz_fluid_command = 'python {} {} {}'.format(os.path.expandvars('$HTR_DIR/scripts/viz_fluid.py'), 
                                             args.json_file,
                                             os.path.join(merge_dir,'*.hdf'))
try:
  subprocess.call(viz_fluid_command, shell=True)
except OSError:
  print("Failed command: {}".format(viz_fluid_command))
  sys.exit()
else: 
  print("Successfully ran command: {}".format(viz_fluid_command))

# Move the generated files to the output directory
try:  
    subprocess.call('mv out_fluid.xmf out*.hdf {}'.format(out_dir), shell=True)
except OSError:  
    print("Failed to move fluid xmf and hdf files to: {}".format(out_dir))
    sys.exit()
else:  
    print("Successfully moved fluid xmf and hdf files to: {}".format(out_dir))

print('')
