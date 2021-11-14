#!/bin/bash

tag="cyclic"
dt=$(date '+%d-%m-%Y-%H:%M:%S')
assignment="a3"
dir="/pvfsmnt/119010115/${assignment}"
job_limit=20
export LD_LIBRARY_PATH=${dir}/build/
mkdir -p ${dir}/tmp
mkdir -p ${dir}/logs
prog="mpi"

log="${dir}/logs/${prog}-${tag}-${dt}.log"
rounds=10
# for i in {1,2,4,8,16,32,64,128}
for i in {1..128}
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
        file="${dir}/tmp/${prog}-${size}-${i}.sh"
        output="${dir}/tmp/${prog}-${size}-${i}.out"
        echo "using ${i} cores with size ${size}"
        echo "#!/bin/bash" > $file
        echo "export LD_LIBRARY_PATH=${dir}/build/" >> $file
        echo "mpirun -n $i ${dir}/build/${prog} ${size} ${rounds} >> ${log}" >> $file
        cat $file
        sbatch --account=csc4005 --partition=debug --qos=normal --nodes=4 --ntasks=$i --output=$output --distribution=${tag} $file
    done
done