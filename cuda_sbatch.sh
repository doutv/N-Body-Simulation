#!/bin/bash

dt=$(date '+%d-%m-%Y-%H:%M:%S')
assignment="a3"
dir="/pvfsmnt/119010115/${assignment}"
export LD_LIBRARY_PATH=${dir}/build/
mkdir -p ${dir}/tmp
mkdir -p ${dir}/logs

prog="cuda"

log="${dir}/logs/${prog}-${dt}.log"
rounds=1
for thread in {1,2,4,8,16,32,64,128,256,512,1024}
do
    for size in {200,1000,5000,10000}
    do
        file="${dir}/tmp/${prog}-${size}-${thread}.sh"
        output="${dir}/tmp/${prog}-${size}-${thread}.out"
        echo "using ${thread} threads with size ${size}"
        echo "#!/bin/bash" > $file
        echo "export LD_LIBRARY_PATH=${dir}/build/" >> $file
        echo "${dir}/build/${prog} ${size} ${rounds} ${thread} >> ${log}" >> $file
        cat $file
        sbatch --wait --account=csc4005 --partition=debug --qos=normal --nodes=1 --ntasks=1 --cpus-per-task=32 --output=$output $file
    done
done
