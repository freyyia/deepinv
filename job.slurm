#!/bin/bash -l
 
################# Part-1 Slurm directives ####################
## Working dir
#SBATCH -D /users/tk2017/code/deepinv
## Environment variables
#SBATCH --export=ALL
## Output and Error Files
#SBATCH -o job-%j.output
#SBATCH -e job-%j.error
## Job name
#SBATCH -J gpu-test
## Run time: "hours:minutes:seconds", "days-hours"
#SBATCH --time=72:00:00
## Memory limit (in megabytes). Total --mem or amount per cpu --mem-per-cpu
#SBATCH --mem-per-cpu=20024
## GPU requirements
#SBATCH --gres gpu:1
## Specify partition
#SBATCH -p gpu

source "${flight_ROOT:-/opt/flight}"/etc/setup.sh
flight env activate conda@base
conda activate ila_exp



echo "$CONDA_PREFIX"


echo "Executing job commands, current working directory is $(pwd)"
echo "I was allocated the following GPU devices: $CUDA_VISIBLE_DEVICES" 

# cd examples/patch-priors
# python demo_patch_priors_CT.py
cd examples/optimization
#python low_dose_ct_experiment.py --prior_type 2 --im_idx 1 --method "MLA" --poisson_level 10 --regularization 4 --step_size 1e-4 --iterations 10000 --inset_loc "(0.52,0.55)"
#python low_dose_ct_experiment.py --prior_type 0 --im_idx 1 --method "MLA" --poisson_level 10 --regularization 4 --step_size 1e-4 --iterations 10000 --inset_loc "(0.52,0.55)"
#python low_dose_ct_experiment.py --prior_type 1 --im_idx 1 --method "MLA" --poisson_level 10 --regularization 4 --step_size 1e-4 --iterations 10000 --inset_loc "(0.52,0.55)"
#python low_dose_ct_experiment.py --prior_type 2 --im_idx 1 --method "SKROCK" --poisson_level 10 --regularization 4 --step_size 5e-5 --iterations 5000 --inset_loc "(0.52,0.55)"
#python low_dose_ct_experiment.py --prior_type 0 --im_idx 1 --method "SKROCK" --poisson_level 10 --regularization 4 --step_size 5e-5 --iterations 5000 --inset_loc "(0.52,0.55)"
#python low_dose_ct_experiment.py --prior_type 1 --im_idx 1 --method "SKROCK" --poisson_level 10 --regularization 4 --step_size 5e-5 --iterations 5000 --inset_loc "(0.52,0.55)"
# python low_dose_ct_experiment.py --prior_type 2 --im_idx 3 --method "MLA" --poisson_level 10 --regularization 4 --step_size 1e-4 --iterations 10000 --inset_loc "(0.42,0.56)"
# python low_dose_ct_experiment.py --prior_type 0 --im_idx 2 --method "MLA" --poisson_level 10 --regularization 4 --step_size 1e-4 --iterations 10000 --inset_loc "(0.42,0.56)"
# python low_dose_ct_experiment.py --prior_type 1 --im_idx 2 --method "MLA" --poisson_level 10 --regularization 4 --step_size 1e-4 --iterations 10000 --inset_loc "(0.42,0.56)"
# python low_dose_ct_experiment.py --prior_type 2 --im_idx 2 --method "SKROCK" --poisson_level 10 --regularization 4 --step_size 5e-5 --iterations 5000 --inset_loc "(0.42,0.56)"
# python low_dose_ct_experiment.py --prior_type 0 --im_idx 2 --method "SKROCK" --poisson_level 10 --regularization 4 --step_size 5e-5 --iterations 5000 --inset_loc "(0.42,0.56)"
# python low_dose_ct_experiment.py --prior_type 1 --im_idx 2 --method "SKROCK" --poisson_level 10 --regularization 4 --step_size 5e-5 --iterations 5000 --inset_loc "(0.42,0.56)"
# python low_dose_ct_experiment.py --prior_type 2 --im_idx 0 --method "MLA" --poisson_level 10 --regularization 4 --step_size 1e-4 --iterations 10000 --inset_loc "(0.45,0.5)"
# python low_dose_ct_experiment.py --prior_type 0 --im_idx 0 --method "MLA" --poisson_level 10 --regularization 4 --step_size 1e-4 --iterations 10000 --inset_loc "(0.45,0.5)"
# python low_dose_ct_experiment.py --prior_type 1 --im_idx 0 --method "MLA" --poisson_level 10 --regularization 4 --step_size 1e-4 --iterations 10000 --inset_loc "(0.45,0.5)"
# python low_dose_ct_experiment.py --prior_type 2 --im_idx 0 --method "SKROCK" --poisson_level 10 --regularization 4 --step_size 5e-5 --iterations 3000 --inset_loc "(0.45,0.5)"
# python low_dose_ct_experiment.py --prior_type 0 --im_idx 0 --method "SKROCK" --poisson_level 10 --regularization 4 --step_size 5e-5 --iterations 3000 --inset_loc "(0.45,0.5)"
# python low_dose_ct_experiment.py --prior_type 1 --im_idx 0 --method "SKROCK" --poisson_level 10 --regularization 4 --step_size 5e-5 --iterations 3000 --inset_loc "(0.45,0.5)"


python low_dose_ct_experiment.py --prior_type 0 --im_idx 1 --method "MLA" --poisson_level 10 --regularization 4 --step_size 1e-4 --iterations 10000 --inset_loc "(0.52,0.55)"
python low_dose_ct_experiment.py --prior_type 0 --im_idx 1 --method "SKROCK" --poisson_level 10 --regularization 4 --step_size 5e-5 --iterations 5000 --inset_loc "(0.52,0.55)"
