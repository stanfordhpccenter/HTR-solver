rm -rf sample* slurm*

for cases in 600; do
   rm -r $cases
   mkdir $cases
   USE_CUDA=0 PROFILE=0 QUEUE="gpu" $HTR_DIR/prometeo.sh -i $cases.json -o $cases
done
