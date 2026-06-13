# Hyperparameter Tuning using WandB Sweeps

Provide a system to automatically optimize the hyperparameters of the neural networks (PPO, SAC, TD3) to improve training efficiency and financial backtest performance. This integrates WandB Sweeps for Bayesian optimization, tracking runs, and early stopping of trials.

## User Review Required

Document anything that requires user review or feedback.

> [!IMPORTANT]
> - WandB Sweeps will run as a separate Python script (`train_with_sweep.py`) on the VPS or local system.
> - The primary optimization metric will be the **Validation Sharpe Ratio** (or validation return) computed over the validation set.
> - Optimization requires a WandB account with the API key configured (already present in the `.env` file).

## Proposed Changes

We will create a new script and config file to manage the hyperparameter sweep.

### [Tuning Engine]

#### [NEW] [sweep_config.yaml](file:///d:/AI-Project/Final/ROBOCHILD/sweep_config.yaml)
Create a WandB sweep configuration YAML file outlining the hyperparameters to search, ranges/values, and optimization objective.
```yaml
name: robochild-hyperparameter-sweep
method: bayes # Bayesian optimization
metric:
  name: val_sharpe
  goal: maximize
parameters:
  # General RL parameters
  learning_rate:
    values: [linear_0.0003, linear_0.00015, constant_0.0003, constant_0.00015, constant_0.00005]
  n_steps:
    values: [1024, 2048, 4096]
  batch_size:
    values: [64, 128, 256, 512]
  gamma:
    values: [0.98, 0.985, 0.99]
  gae_lambda:
    values: [0.92, 0.95, 0.98]
  
  # PPO specific
  vf_coef:
    values: [0.5, 0.8, 1.2]
  ent_coef:
    values: [0.01, 0.015, 0.02, 0.03]
  clip_range:
    values: [0.2, 0.25, 0.3]
    
  # Ensemble weights
  ppo_weight:
    values: [0.4, 0.5, 0.6]
  sac_weight:
    values: [0.2, 0.3, 0.4]
  td3_weight:
    values: [0.1, 0.2, 0.3]
```

#### [NEW] [train_with_sweep.py](file:///d:/AI-Project/Final/ROBOCHILD/train_with_sweep.py)
Create the main execution script that WandB Sweep agent will run. It will:
- Parse hyperparameters from `wandb.config`
- Run training on `FuturesTradingEnv` using the specified hyperparameters
- Evaluate the model on validation data, computing return, win rate, and Sharpe Ratio
- Log metrics to WandB to drive the optimization sweep

## Verification Plan

### Automated Tests
- We will test the sweep setup locally using a short trial run (e.g. 5,000 steps) to verify that parameters are correctly parsed, model trains, validation is evaluated, and results are logged to WandB.
```powershell
python train_with_sweep.py --test-run
```
- Launch a WandB sweep agent locally to ensure the controller initiates runs correctly.
