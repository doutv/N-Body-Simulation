#!/bin/bash

tag="cyclic"
dt=$(date '+%d-%m-%Y-%H:%M:%S')
assignment="a3"
dir="/pvfsmnt/119010115/${assignment}"
job_limit=2
export LD_LIBRARY_PATH=${dir}/build/
mkdir -p ${dir}/tmp
mkdir -p ${dir}/logs

read -p "Enter program (mpi or pthread): " prog
if [ "$prog" = "mpi" ]
then
    log="${dir}/logs/${prog}-${tag}-${dt}.log"
    # for i in {1,2,4,8,16,32,64,128}
    for i in {1..128}
    do
        for size in {200,400,800}
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
            if [ "$prog" = "mpi" ]; then 
                echo "mpirun -n $i ${dir}/build/${prog} ${size} >> ${log}" >> $file
            else
                echo "${dir}/build/${prog} ${size} >> ${log}" >> $file
            fi;
            cat $file
            sbatch --account=csc4005 --partition=debug --qos=normal --nodes=4 --ntasks=$i --output=$output --distribution=${tag} $file
        done
    done
else
    log="${dir}/logs/${prog}-${tag}-${dt}.log"
    for i in {1..32}
    do
        for size in {200,400,800}
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
            if [ "$prog" = "mpi" ]; then 
                echo "mpirun -n $i ${dir}/build/${prog} ${size} >> ${log}" >> $file
            else
                echo "${dir}/build/${prog} ${size} ${i} >> ${log}" >> $file
            fi;
            cat $file
            sbatch --account=csc4005 --partition=debug --qos=normal --nodes=1 --ntasks=32 --output=$output $file
        done
    done
fi