#!/bin/bash


n=500
parallelism=16 # number of trainings/runs that are run in parallel
output_dir=$(pwd)"/data" # output dir

agent_dir='../' 

mkdir -p "${output_dir}"

cd "$agent_dir"

flags_train=""
flags_run=""
filename="agent"
train=true
run=true
while [ "$1" != "" ]; do
  case $1 in
    -d | --dqn )
      flags_train="${flags_train} --dqn"
      flags_run="${flags_train} --dqn"
      filename="${filename}_dqn"
      ;;
    -dd | --ddqn )
      flags_train="${flags_train} --ddqn"
      flags_run="${flags_train} --ddqn"
      filename="${filename}_ddqn"
      ;;
    -t | --train-only | --no-run )
      run=false
      ;;
    -r | --run-only | --no-train )
      train=false
      ;;
    -o | --overwrite )
      flags_train="${flags_train} --overwrite"
      flags_run="${flags_run} --overwrite"
      ;;
    * )
      echo "unknown command line argument ${1}"
      ;;
  esac
  shift
done

run_command () {
  local i=$1
  #
  if [ "$train" = true ] ; then
    echo $(date +"%Y-%m-%d %T") " Training agent ${filename}_${i}"
    # Run the command with the current suffix in the filename
    python train_agent.py --f "${output_dir}/${filename}_${i}" $flags_train
  fi
  #
  if [ "$run" = true ] ; then
    echo $(date +"%Y-%m-%d %T") " Running agent ${filename}_${i}"
    # Run the command with the current suffix in the filename
    python run_agent.py --f "${output_dir}/${filename}_${i}" $flags_run
  fi
}

for ((i=1; i <= $n; i++)); do

  while (( $(jobs -r -p | wc -l) >= $parallelism )); do
    sleep 1
  done
  #
  run_command "$i" &
  #
done


wait