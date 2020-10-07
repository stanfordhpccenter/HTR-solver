rm -rf sample* slurm*

for cases in 100; do
   rm -r $cases
   mkdir $cases
   USE_CUDA=1 PROFILE=0 QUEUE="gpu" $HTR_DIR/jobscripts/prometeo.sh -i $cases.json -o $cases
done
