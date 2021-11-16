#!/bin/bash

dt=$(date '+%d-%m-%Y-%H:%M:%S')
assignment="a3"
dir="/pvfsmnt/119010115/${assignment}"
job_limit=30
export LD_LIBRARY_PATH=${dir}/build/
mkdir -p ${dir}/tmp
mkdir -p ${dir}/logs

prog="cuda"

log="${dir}/logs/${prog}-${dt}.log"
rounds=10
for thread in {1..32}
do
    for size in {200,1000,5000,10000}
    do
        line=$(squeue --me | wc -l)
        while [ $line -ge $job_limit ]
        do
            line=$(squeue --me | wc -l) 
            echo "$line jobs in squeue"
            sleep 1s
        done
        file="${dir}/tmp/${prog}-${size}-${thread}.sh"
        output="${dir}/tmp/${prog}-${size}-${thread}.out"
        echo "using ${thread} threads with size ${size}"
        echo "#!/bin/bash" > $file
        echo "export LD_LIBRARY_PATH=${dir}/build/" >> $file
        echo "${dir}/build/${prog} ${size} ${rounds} ${thread} >> ${log}" >> $file
        cat $file
        sbatch --wait --account=csc4005 --partition=debug --qos=normal --nodes=1 --ntasks=1 --output=$output $file
    done
done