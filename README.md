# YOLO Dataset and Training Overview

## Dataset Downloads

- **Main dataset**:  
  [Download Link](https://ehb-my.sharepoint.com/:f:/g/personal/fabian_serkeyn_student_ehb_be/EgpgoJ1-8wlGlwKK60WjJhkBKVuXYIbkiiEJvezx1ySMoQ?e=s7G5lP)  
  Password: `yolodataset`  
  (Also added locally on the workstation)

- **Local copy (workstation)**:  
  [Local Data Folder](https://ehb-my.sharepoint.com/:f:/g/personal/fabian_serkeyn_student_ehb_be/EsBCTAUhT3tGm5zmW-cOECIBaKuK9AMyiIRDCS_0UvEgBg?e=og6mEU)

## File and Folder Structure

- `yolo11-fine/`  
  Contains all training runs including:
  - Tracked results
  - Performance metrics
  - Model checkpoints
  - Logs

- `fine-dataset.yaml`  
  Required YAML configuration file for training. Defines:
  - Dataset paths
  - Class names
  - Number of classes

- `main.py`  
  Main script for training. Contains all adjustable training parameters.

- `training_runs.xlsx`  
  Spreadsheet with basic training log entries and notes.
