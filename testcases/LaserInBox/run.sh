
# Note: 500 iterations of this test case (6 dumps) requires 1.6GB of disk space, including 
# files generated during post-processing.

# Paramters
input=LaserInBox.json
outdir=data

# Check
if [ -d $outdir ]; then
   echo "$outdir already exists.  Aborting to avoid overwrite."
   return;
fi
mkdir $outdir
cp $input $outdir/. # For safe keeping

# Set up environment
export RESERVED_CORES=8 # For Lasses
export USE_CUDA=1 # Only with GPUs
export PROFILE=0
export QUEUE="pdebug"

# Run
$HTR_DIR/prometeo.sh -i $input -o $outdir
