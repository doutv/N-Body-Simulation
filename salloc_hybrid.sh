#!/bin/bash

tag="salloc"
dt=$(date '+%d-%m-%Y-%H:%M:%S')
assignment="a3"
dir="/pvfsmnt/119010115/${assignment}"
export LD_LIBRARY_PATH=${dir}/build/
mkdir -p ${dir}/tmp
mkdir -p ${dir}/logs

prog="hybrid"

log="${dir}/logs/${prog}-${tag}-${dt}.log"
rounds=1
task=4
for thread in {1..32}
do
    for size in {200,1000,5000,10000}
    do
        echo "writing to ${log}"
        echo "using ${task} tasks and ${thread} threads with size ${size}"
        mpirun -n ${task} ${dir}/build/${prog} ${size} ${rounds} ${thread} >> ${log}
    done
done