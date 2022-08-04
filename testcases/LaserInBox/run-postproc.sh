
name=LaserInBox
dir="data" # corresponds to outdir in run.sh
startIter=0
skipIter=100 # if zero, then process all available iterations
stopIter=500

# This generates (1) input file (json), (2) initial solution files (hdf).
if [[ $skipIter -eq 0 ]]; then
   python3 postProc.py $name $dir
else
   python3 postProc.py $name $dir --start $startIter --skip $skipIter --stop $stopIter
fi
