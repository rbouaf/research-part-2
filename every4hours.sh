#!/bin/bash

while true; do
  echo "Running scheduled job at $(date)" >> /home/2022/rbouaf/data/cron_log.log

  # Your original script logic
  source /home/2022/rbouaf/.bashrc
  conda activate pytorch_1_3_1_env
  module load slurm
  sbatch /home/2022/rbouaf/data/slurm_all.sh
  squeue --me >> /home/2022/rbouaf/data/cron_log.log 2>&1

  # Wait for 4 hours
  sleep 4h
done

