rsync -a ./ slurm:/pvfsmnt/119010115/a3 -v --exclude-from=exclude.list
# rsync -a ./gen_sbatch.sh slurm:/pvfsmnt/119010115/a3/ -v
ssh -t slurm "chmod +x /pvfsmnt/119010115/a3/gen_sbatch.sh"
ssh slurm
