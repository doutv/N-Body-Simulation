rsync -a ./ slurm:/pvfsmnt/119010115/a3 -v --exclude-from=exclude.list
rsync -a ./*_sbatch.sh slurm:/pvfsmnt/119010115/a3 -v
ssh -t slurm "chmod +x /pvfsmnt/119010115/a3/*_sbatch.sh"
ssh slurm