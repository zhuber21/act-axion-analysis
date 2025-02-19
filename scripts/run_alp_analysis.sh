#!/bin/bash -l
# For running ALP analysis on NERSC Perlmutter
# Submit this script as: "./run_alp_analysis.sh" instead of "sbatch run_alp_analysis.sh"

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

export RUN_TAG=test_f150_run_20250212   # CHANGE THIS FOR EACH RUN!!

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
#SBATCH --time 00:15:00
#SBATCH --nodes=13
#SBATCH --ntasks-per-node=6
#SBATCH --output=/pscratch/sd/z/zbh5/results/$RUN_TAG.out

srun -n 78 -c 32 --cpu_bind=cores python3 get_depth1_angle_parallel.py dr6_depth1_ps_config.yaml $RUN_TAG

# Call collect_npy_files.py to group all output npy files into a single npy file
python3 collect_npy_files.py /pscratch/sd/z/zbh5/results/angle_calc_$RUN_TAG/
# Amend path for output result copying location to match job name
cp -r /pscratch/sd/z/zbh5/results/angle_calc_$RUN_TAG/ /global/homes/z/zbh5/alp_outputs/angle_calc_$RUN_TAG/
cp /pscratch/sd/z/zbh5/results/$RUN_TAG.out /global/homes/z/zbh5/alp_outputs/angle_calc_$RUN_TAG/
EOF

# Now submit the batch job
sbatch prepare-env.sl
