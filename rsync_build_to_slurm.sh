dst="/pvfsmnt/119010115/a3"
cmake --build ./build -j6
rsync -a ./build slurm:${dst} -v
rsync -a ./*_sbatch.sh slurm:${dst}/build -v
ssh -t slurm "chmod +x ${dst}/build/*.sh"
ssh slurm