# CSCE479-Homeworks

Steps to run homework:
1. Connect to Swan server: `https://swan-ood.unl.edu/`
2. Interactive Apps ==> Code Server or Jupyter Lab
3. cd into desired HW directory
4. Run `sbatch ~/submit_gpu.sh python main.py`
5. Access newly created `slurm` file

`submit_gpu.sh`
```
#!/bin/sh

#SBATCH --time=6:00:00                        # Maximum run time in hh:mm:ss

#SBATCH --mem=16000                           # Maximum memory required (in megabytes)

#SBATCH --job-name=hackathon3-test                # Job name (to track progress)   

#SBATCH --partition=csce_gpu,csce_gpu_preempt # Partition on which to run job 

#SBATCH --gres=gpu:1                          # Don't change this, it requests a GPU

#SBATCH --constraint=gpu_16gb                 # will request a GPU with 16GB of RAM, independent of the type of card



module load mamba

conda activate /mnt/nrdstor/cse479/shared/envs

$@
```
