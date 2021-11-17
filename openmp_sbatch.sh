#!/bin/bash

dt=$(date '+%d-%m-%Y-%H:%M:%S')
assignment="a3"
dir="/pvfsmnt/119010115/${assignment}"
export LD_LIBRARY_PATH=${dir}/build/
mkdir -p ${dir}/tmp
mkdir -p ${dir}/logs

prog="openmp"
rounds=1
log="${dir}/logs/${prog}-${dt}.log"
for thread in {33..64}
do
    realthread=32
    for size in {200,1000,5000,10000}
    do
        file="${dir}/tmp/${prog}-${size}-${thread}.sh"
        output="${dir}/tmp/${prog}-${size}-${thread}.out"
        echo "using ${thread} threads with size ${size}"
        echo "#!/bin/bash" > $file
        echo "export LD_LIBRARY_PATH=${dir}/build/" >> $file
        echo "${dir}/build/${prog} ${size} ${rounds} ${thread} >> ${log}" >> $file
        cat $file
        sbatch --wait --account=csc4005 --partition=debug --qos=normal --nodes=1 --ntasks=1 --cpus-per-task=${realthread} --output=$output $file
    done
done
