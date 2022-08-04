#!/bin/bash -eu

###############################################################################
# Helper functions
###############################################################################

function quit {
   echo "$1" >&2
   exit 1
}

###############################################################################
# Derived options
###############################################################################

# Compute unique job ID
if [[ ! -z "${PBS_JOBID:-}" ]]; then
   JOBID="$PBS_JOBID"
elif [[ ! -z "${SLURM_JOBID:-}" ]]; then
   JOBID="$SLURM_JOBID"
elif [[ ! -z "${LSB_JOBID:-}" ]]; then
   JOBID="$LSB_JOBID"
else
   JOBID="$(date +%s)"
fi

# Create a directory on the scratch filesystem, for output (if $SCRATCH is
# defined, and the user hasn't specified an output directory explicitly).
OUT_DIR=
OUT_DIR_FOLLOWS=false
for ARG in $ARGS; do
   if [[ "$OUT_DIR_FOLLOWS" == true ]]; then
      OUT_DIR="$ARG"
      break
   elif [[ "$ARG" == "-o" ]]; then
      OUT_DIR_FOLLOWS=true
   fi
done
if [[ -z "$OUT_DIR" ]]; then
   if [[ ! -z "${SCRATCH:-}" ]]; then
      OUT_DIR="$SCRATCH"/"$JOBID"
      mkdir "$OUT_DIR"
      ARGS="$ARGS -o $OUT_DIR"
   else
      OUT_DIR=.
   fi
fi
if [[ "$OUT_DIR" != "${OUT_DIR%[[:space:]]*}" ]]; then
   quit "Cannot handle spaces in output directory"
fi
echo "Sending output to $OUT_DIR"

# Prepare Legion configuration
CORES_PER_RANK=$(( CORES_PER_NODE / RANKS_PER_NODE ))
RAM_PER_RANK=$(( RAM_PER_NODE / RANKS_PER_NODE ))
THREADS_PER_RANK=$(( CORES_PER_RANK - RESERVED_CORES ))
THREADS_PER_NUMA=$(( THREADS_PER_RANK / NUMA_PER_RANK ))
if (( CORES_PER_NODE < RANKS_PER_NODE ||
      CORES_PER_NODE % RANKS_PER_NODE != 0 ||
      RESERVED_CORES >= CORES_PER_RANK )); then
   quit "Cannot split $CORES_PER_NODE core(s) into $RANKS_PER_NODE rank(s)"
fi

# Increase number of reserved cores if we cannot split them for each numa domain
while (( THREADS_PER_RANK < NUMA_PER_RANK ||
         THREADS_PER_RANK % NUMA_PER_RANK != 0)); do
   RESERVED_CORES=$(( RESERVED_CORES + 1 ))
   THREADS_PER_RANK=$(( CORES_PER_RANK - RESERVED_CORES ))
   THREADS_PER_NUMA=$(( THREADS_PER_RANK / NUMA_PER_RANK ))
   if (( RESERVED_CORES >= CORES_PER_RANK )); then
      quit "Cannot find a suitable combination of RESERVED_CORES and CORES_PER_RANK"
   fi
done

if (( THREADS_PER_RANK < NUMA_PER_RANK ||
      THREADS_PER_RANK % NUMA_PER_RANK != 0)); then
   quit "Cannot split $THREADS_PER_RANK thread(s) into $NUMA_PER_RANK openmp proc(s)"
fi
if [[ "$USE_CUDA" == 1 ]]; then
   GPUS_PER_RANK=$(( GPUS_PER_NODE / RANKS_PER_NODE ))
   if (( GPUS_PER_NODE < RANKS_PER_NODE ||
         GPUS_PER_NODE % RANKS_PER_NODE != 0 )); then
      quit "Cannot split $GPUS_PER_NODE GPU(s) into $RANKS_PER_NODE rank(s)"
   fi
fi

# Define default number of threads dedicated to the runtime
IO_THREADS="${IO_THREADS:-1}"
UTIL_THREADS="${UTIL_THREADS:-2}"
# if we are running with openMP we need to reserve 1 core for serial tasks
BGWORK_THREADS="${BGWORK_THREADS:-$(( RESERVED_CORES - UTIL_THREADS - USE_OPENMP ))}"
if (( BGWORK_THREADS < 0 )); then
   quit "You did not reserve enough cores for the runtime"
fi

# Add debugging flags
DEBUG_OPTS=
if [[ "$DEBUG" == 1 ]]; then
   DEBUG_OPTS="-ll:force_kthreads -logfile $OUT_DIR/%.log -lg:safe_ctrlrepl 2"
fi
# Add profiling flags
PROFILER_OPTS=
if [[ "$PROFILE" == 1 ]]; then
   PROFILER_OPTS="-lg:prof $NUM_RANKS -lg:prof_logfile $OUT_DIR/prof_%.log"
fi
# Add CUDA options
CPU_OPTS=
if [[ "$USE_OPENMP" == 1 ]]; then
   CPU_OPTS="-ll:cpu 1 -ll:ocpu $NUMA_PER_RANK -ll:onuma 1 -ll:othr $THREADS_PER_NUMA -ll:ostack 8"
else
   CPUS=$(( CORES_PER_RANK - RESERVED_CORES ))
   CPU_OPTS="-ll:cpu $CPUS"
fi
# Add CUDA options
GPU_OPTS=
if [[ "$USE_CUDA" == 1 ]]; then
   GPU_OPTS="-ll:gpu $GPUS_PER_RANK -ll:fsize $FB_PER_GPU -ll:zsize 512 -ll:ib_zsize 512"
fi
# Add GASNET options
GASNET_OPTS=
if [[ "$LOCAL_RUN" == 0 ]]; then
   GASNET_OPTS="-ll:rsize 512 -ll:ib_rsize 512 -ll:gsize 0"
fi
# Synthesize final command
COMMAND="$EXECUTABLE $ARGS \
   $DEBUG_OPTS $PROFILER_OPTS \
   $CPU_OPTS \
   $GPU_OPTS \
   -ll:util $UTIL_THREADS -ll:io $IO_THREADS -ll:bgwork $BGWORK_THREADS \
   -ll:cpu_bgwork 100 -ll:util_bgwork 100 \
   -ll:csize $RAM_PER_RANK \
   $GASNET_OPTS \
   -ll:stacksize 8 -lg:sched -1 -lg:hysteresis 0"
echo "Invoking Legion on $NUM_RANKS rank(s), $NUM_NODES node(s) ($RANKS_PER_NODE rank(s) per node), as follows:"
echo $COMMAND
