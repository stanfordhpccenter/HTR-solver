rm -rf sample* slurm*

for cases in 100; do
   rm -r $cases
   mkdir $cases
   PROFILE=0 $HTR_DIR/prometeo.sh -i $cases.json -o $cases
done
