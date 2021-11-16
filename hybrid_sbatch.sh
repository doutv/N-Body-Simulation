#!/bin/bash

tag="block"
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
        file="${dir}/tmp/${prog}-${size}-${task}-${thread}.sh"
        output="${dir}/tmp/${prog}-${size}-${task}-${thread}.out"
        echo "using ${task} tasks and ${thread} threads with size ${size}"
        echo "#!/bin/bash" > $file
        echo "export LD_LIBRARY_PATH=${dir}/build/" >> $file
        echo "mpirun -n ${task} ${dir}/build/${prog} ${size} ${rounds} ${thread} >> ${log}" >> $file
        echo "sbatch file: ${file}"
        cat $file
        sbatch --wait --account=csc4005 --partition=debug --qos=normal --ntasks=${task} --cpus-per-task=${thread} --output=$output $file
    done
done