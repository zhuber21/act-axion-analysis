#!/bin/bash -l
# For running null tests for ALP timestream analysis on NERSC Perlmutter
# Submit this script as: "./run_null_test.sh" instead of "sbatch run_null_test.sh"

# Prepare user env needed for Slurm batch job
# such as module load, setup runtime environment variables, or copy input files, etc.
# Basically, these are the commands you usually run ahead of the srun command

conda activate /global/common/software/act/zhuber/alp_mpi_env
module load cray-mpich-abi

# I believe this should be the same as srun's number of logical cpus
# when OMP_PLACES=threads - this ensures the FFTs can use all threads
export OMP_NUM_THREADS=32
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

export RUN_TAG=temporal_Feb2019_20250716   # CHANGE THIS FOR EACH RUN!!

# Generate the Slurm batch script below with the here document,
# then when sbatch the script later, the user env set up above will run on the login node
# instead of on a head compute node (if included in the Slurm batch script),
# and inherited into the batch job.

cat << EOF > prepare-env.sl
#!/bin/bash
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH --account=mp107b
#SBATCH --mail-user=zbh5@cornell.edu
#SBATCH --mail-type=ALL
#SBATCH -J $RUN_TAG
#SBATCH --time 00:30:00
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=100
#SBATCH --output=/pscratch/sd/z/zbh5/results/null_tests/$RUN_TAG.out

srun -n 500 -c 32 --cpu_bind=cores python3 get_amp_nul_test.py null_test_config.yaml $RUN_TAG

# Call collect_null_test_npy_files.py to group all output npy files for the splits of the sims into single npy files
python3 collect_null_test_npy_files.py /pscratch/sd/z/zbh5/null_tests/null_test_results_$RUN_TAG/
# Amend path for output result copying location to match job name
cp -r /pscratch/sd/z/zbh5/null_tests/null_test_results_$RUN_TAG/ /global/homes/z/zbh5/null_test_outputs/null_test_results_$RUN_TAG/
cp /pscratch/sd/z/zbh5/null_tests/$RUN_TAG.out /global/homes/z/zbh5/null_test_outputs/null_test_results_$RUN_TAG/
EOF

# Now submit the batch job
sbatch prepare-env.sl
