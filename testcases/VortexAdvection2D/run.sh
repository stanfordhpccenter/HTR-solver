rm -rf sample* slurm*

python scale_up.py base.json -n $1

Np=16
for i in `seq 1 $1`;
do
   rm -rf  $Np
   mkdir $Np
   USE_CUDA=1 PROFILE=0 QUEUE="gpu" $HTR_DIR/prometeo.sh -i VortexAdvection2D_$Np.json -o $Np
   Np=$((Np*2))
done
