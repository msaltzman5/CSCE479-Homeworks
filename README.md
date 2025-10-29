# CSCE479-Homeworks

cd into desired HW directory.
Run `sbatch ~/submit_gpu.sh python main.py`

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
