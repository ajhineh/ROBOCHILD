# WandB Advanced Integration & Model Registry Plan

This updated implementation plan integrates the research team's feedback to unlock the full potential of Weights & Biases (WandB) for training monitoring, model registry, artifact versioning, and advanced backtesting diagnostics.

---

## 📋 Prioritized Implementation Roadmap

### 1. High Priority (Immediate Execution)

#### A. Smart Artifact Tagging & Aliases
- **[trainer.py](file:///d:/AI-Project/Final/ROBOCHILD/src/agent/trainer.py)**:
  - When a training phase completes, package the model weights and `VecNormalize` stats.
  - Tag the artifact using versions (`v1`, `v2`, etc.) and programmatically add aliases (`best`, `stable`) based on performance thresholds:
    ```python
    artifact = wandb.Artifact(f"ensemble_model_{symbol}", type="model")
    artifact.add_file(model_zip_path)
    artifact.add_file(vec_norm_path)
    # Register aliases based on metrics
    artifact.add_alias("best")
    run.log_artifact(artifact)
    ```

#### B. Full Backtest Table & Plotly Chart Logging
- **[training_evaluator.py](file:///d:/AI-Project/Final/ROBOCHILD/src/analysis/training_evaluator.py)**:
  - Construct a `wandb.Table` containing the complete list of trades (Entry time, Exit time, Action, PnL per trade, Execution Price, Slippage, Fees).
  - Generate and log interactive Plotly figures (Equity Curve progression, Drawdown curves, Action Distribution histogram) directly to the WandB run dashboard.

#### C. Dedicated Sweep Configuration Structure
- **[NEW] [sol_sweep.yaml](file:///d:/AI-Project/Final/ROBOCHILD/sweeps/sol_sweep.yaml)**:
  - Create a dedicated subdirectory `sweeps/` and place `sol_sweep.yaml` inside it.
  - Set the search strategy to Bayesian Optimization to run 30-50 trials maximizing `val_eval_mean_reward` and `ppo_explained_variance`.

---

### 2. Medium Priority

#### A. Model Registry & Production Promotion
- Promote the winner model of the sweep to the **WandB Model Registry** under a registered model namespace (e.g., `ROBOCHILD-SOL-Production`).
- **[model_loader.py](file:///d:/AI-Project/Final/ROBOCHILD/src/core/rl_shared/model_loader.py)**:
  - Configure the model loader to pull the latest production-promoted model directly from the Registry (using the tag `:production`) instead of arbitrary local filenames.

#### B. Smart Alerts & Monitoring
- **[trainer.py](file:///d:/AI-Project/Final/ROBOCHILD/src/agent/trainer.py)**:
  - Setup threshold triggers in training callbacks. Send a `wandb.alert()` notification if:
    - `ppo_explained_variance` < 0.4 after 50,000 steps.
    - Mean reward stays negative for more than 100,000 steps.
    - KL Divergence > 0.05 (indicates policy update is too aggressive / unstable).

---

## 📂 Proposed File Changes

### [Component: sweeps/]
#### [NEW] [sol_sweep.yaml](file:///d:/AI-Project/Final/ROBOCHILD/sweeps/sol_sweep.yaml)
Define the Bayesian search hyperparameter grid for SOL.

---

### [Component: src/agent/]
#### [MODIFY] [trainer.py](file:///d:/AI-Project/Final/ROBOCHILD/src/agent/trainer.py)
Add programmatic alert triggers inside `ProgressCallback` / training loops, and implement smart artifact logging with aliases.

---

### [Component: src/analysis/]
#### [MODIFY] [training_evaluator.py](file:///d:/AI-Project/Final/ROBOCHILD/src/analysis/training_evaluator.py)
Convert backtest trades to `wandb.Table` and log Plotly interactive charts. Replace hardcoded project routing.

---

### [Component: src/core/rl_shared/]
#### [MODIFY] [model_loader.py](file:///d:/AI-Project/Final/ROBOCHILD/src/core/rl_shared/model_loader.py)
Integrate Model Registry model downloading with a robust local backup fallback path.

---

## 🧪 Verification Plan

### Automated Tests
- Validate code updates locally with `python validate_pipeline.py`.

### Manual Verification
- Trigger a mock training run and verify on the WandB dashboard that:
  - Trade tables are logged under the "Files/Tables" tab.
  - Model Artifact versions and aliases are properly registered under the "Artifacts" tab.
  - Check that alerts are correctly triggered when simulating an unstable model run (e.g. using very high learning rates).
