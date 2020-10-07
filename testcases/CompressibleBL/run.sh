rm -rf sample* slurm*

#LEGION_FREEZE_ON_ERROR=1 
USE_CUDA=1 PROFILE=0 QUEUE="gpu" $HTR_DIR/prometeo.sh -i CBL.json
#USE_CUDA=0 PROFILE=0 QUEUE="all" $HTR_DIR/prometeo.sh -i CBL.json
