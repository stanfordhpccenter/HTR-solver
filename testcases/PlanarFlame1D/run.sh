
# Generate input files
python3 MakeInput.py base.json

# For Lassen
export RESERVED_CORES=8

# Number of scenarios to run (each with a different equivalence ratio, phi)
nPhi=4

for i in `seq 1 $nPhi`;
do
   input="PlanarFlame1D-$i.json"
   outdir="out-$i"
   rm -rf $outdir
   mkdir $outdir
   cp $input $outdir/.
   
   # Run HTR
   USE_CUDA=1 PROFILE=0 QUEUE="pbatch" $HTR_DIR/prometeo.sh -i $input -o $outdir
done
