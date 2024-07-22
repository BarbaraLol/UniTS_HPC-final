#!/bin/bash

#SBATCH -A dssc
#SBATCH -p THIN
#SBATCH --job-name=latency
#SBATCH --nodes=2              # Requesting two nodes
#SBATCH --exclusive
#SBATCH --ntasks=2             # Two tasks to ensure one task per node
#SBATCH --time=02:00:00
#SBATCH --exclude=fat[001-002]

# Load the openMPI module
module load openMPI/4.1.5/gnu
export OMPI_MCA_pml=ucx

# Define variables for the naming of files and stuff
dt=$(date +"%Y%m%d_%H%M%S")
operazione="latenza"
partizione="magra"

# Define filepaths
src_path="/u/dssc/arusso/osu-micro-benchmarks-7.3/c/mpi/pt2pt/standard/osu_latency"
out_file="results_${operazione}/${operazione}_${partizione}_${dt}.csv"

# Create the output directory if it doesn't exist
output_dir=$(dirname "$out_file")
mkdir -p "$output_dir"

# Create the CSV file with header
echo "CPU_Pair,MessageSize,Latency" > $out_file

# Iterate over all possible CPU pairs starting with core 0
for cpu in {1..47}
echo "Running latency test for CPU pair 0,$cpu..."

# Run the mpirun command and capture the output
mpirun -np 2 --map-by ppr:1:node --bind-to core ${src_path} -i 1000 -x 100 > temp_output.txt

# Extract and format the relevant data, filter out unwanted lines
tail -n +3 temp_output.txt | grep -v '^#' | awk -v cpu_pair="0,1" '{print cpu_pair "," $1 "," $2}' >> $out_file


# Clean up the temporary output file
rm temp_output.txt
