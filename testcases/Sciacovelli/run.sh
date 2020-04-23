rm -rf sample0 slurm*

USE_CUDA=1 PROFILE=0 QUEUE="gpu" $HTR_DIR/src/prometeo.sh -i ChannelFlow.json #-lg:inorder
