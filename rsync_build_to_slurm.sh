cmake --build ./build -j6
rsync -a ./build slurm:/pvfsmnt/119010115/a3 -v
rsync -a ./gen_sbatch.sh slurm:/pvfsmnt/119010115/a3/ -v
ssh -t slurm "chmod +x /pvfsmnt/119010115/a3/gen_sbatch.sh"
ssh slurm