#!/bin/bash -eu

###############################################################################
# Inputs
###############################################################################

# Which group to submit jobs under (if a scheduler is available)
export GROUP="${GROUP:-}"

# Which queue/partition to use (if a scheduler is available)
export QUEUE="${QUEUE:-}"

# Which job to wait for before starting (if a scheduler is available)
export AFTER="${AFTER:-}"

# Whether to use GPUs (if available)
export USE_CUDA="${USE_CUDA:-1}"

# Whether to emit Legion profiler logs
export PROFILE="${PROFILE:-0}"

# Whether to print a backtrace on crash (interferes with signal handling)
export REALM_BACKTRACE="${REALM_BACKTRACE:-0}"

# Whether to freeze Legion execution on crash
export DEBUG="${DEBUG:-0}"

# How many ranks to instantiate per node
export RANKS_PER_NODE="${RANKS_PER_NODE:-1}"

# How many cores per rank to reserve for the runtime
export RESERVED_CORES="${RESERVED_CORES:-4}"

# Whether this is a job issued by the CD/CI system
export CI_RUN="${CI_RUN:-0}"

###############################################################################
# Helper functions
###############################################################################

function quit {
    echo "$1" >&2
    exit 1
}

function read_json {
    python3 -c "
import json
def wallTime(sample):
  return int(sample['Mapping']['wallTime'])
def numRanks(sample):
  tiles = sample['Mapping']['tiles']
  tilesPerRank = sample['Mapping']['tilesPerRank']
  xRanks = int(tiles[0]) / int(tilesPerRank[0])
  yRanks = int(tiles[1]) / int(tilesPerRank[1])
  zRanks = int(tiles[2]) / int(tilesPerRank[2])
  return int(xRanks * yRanks * zRanks)
def mixture(sample):
  assert sample['Flow']['mixture']['type'] == 'ConstPropMix'    or \
         sample['Flow']['mixture']['type'] == 'IsentropicMix'   or \
         sample['Flow']['mixture']['type'] == 'AirMix'          or \
         sample['Flow']['mixture']['type'] == 'CH41StMix'       or \
         sample['Flow']['mixture']['type'] == 'CH4_30SpMix'     or \
         sample['Flow']['mixture']['type'] == 'CH4_43SpIonsMix' or \
         sample['Flow']['mixture']['type'] == 'CH4_26SpIonsMix' or \
         sample['Flow']['mixture']['type'] == 'FFCM-1Mix'       or \
         sample['Flow']['mixture']['type'] == 'BoivinMix'       or \
         sample['Flow']['mixture']['type'] == 'H2_UCSDMix'
  return sample['Flow']['mixture']['type']
f = json.load(open('$1'))
if '$2' == 'single':
  print('{} {} {}'.format(wallTime(f), numRanks(f), mixture(f)))
elif '$2' == 'dual':
  assert mixture(f['configs'][0]) == mixture(f['configs'][1])
  print('{} {} {}'.format(max(wallTime(f['configs'][0]), wallTime(f['configs'][1])), \
                          numRanks(f['configs'][0]) + numRanks(f['configs'][1]), \
                          mixture(f['configs'][0])))
else:
  assert(false)"
}

###############################################################################
# Derived options
###############################################################################

# Total wall-clock time is the maximum across all samples.
# Total number of ranks is the sum of all sample rank requirements.
MINUTES=0
NUM_RANKS=0
MIX=""
function parse_config {
    read -r _MINUTES _NUM_RANKS _MIX <<<"$(read_json "$@")"
    MINUTES=$(( MINUTES > _MINUTES ? MINUTES : _MINUTES ))
    NUM_RANKS=$(( NUM_RANKS + _NUM_RANKS ))
    MIX=$_MIX
}
for (( i = 1; i <= $#; i++ )); do
    j=$((i+1))
    if [[ "${!i}" == "-i" ]] && (( $i < $# )); then
        parse_config "${!j}" "single"
    elif [[ "${!i}" == "-m" ]] && (( $i < $# )); then
        parse_config "${!j}" "dual"
    fi
done
if (( NUM_RANKS < 1 )); then
    quit "No configuration files provided"
fi
export MINUTES
export NUM_RANKS
export EXECUTABLE="$HTR_DIR"/src/prometeo_"$MIX".exec

###############################################################################

source "$HTR_DIR"/jobscripts/run.sh
