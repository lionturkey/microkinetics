# RL Point Kinetics Microreactor Environment

To create the conda environment:

```
conda create -n micro python=3.11
conda activate micro
conda install stable-baselines3 gymnasium matplotlib imageio numpy tensorboard
```


To replicate results with run named "replicant" start by running `python micro_runner replicant`. This will start the training process and save everything in runs/replicant. If you wish to view run progress in tensorboard, run `tensorboard --logdir runs/replicant/logs`. After training finished, see how the trained model performs with `python micro_runner replicant --run_type load`. To see how it performs with a longer transient, run `python micro_runner replicant --run_type load --profile longtest`.

This mostly covers the cases in our paper, but other options can be found at the bottom of micro_runner.py.
