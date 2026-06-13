# WandB Advanced Integration & Model Registry Plan (V2 - Approved)

This document represents the finalized, approved implementation plan (V2) for integrating Weights & Biases (WandB) advanced capabilities including Registry, Artifacts, Sweeps, Alerts, and automated Reports.

---

## 📋 Prioritized Implementation Roadmap

### 1. High Priority (Immediate Execution)

#### A. Smart Artifact Tagging & Model Registry Promotion
- **[trainer.py](file:///d:/AI-Project/Final/ROBOCHILD/src/agent/trainer.py)**:
  - Artifacts will be created at the end of training phases.
  - In addition to standard tagging, models will be promoted to the WandB Model Registry under a unified schema with the `:candidate` alias (indicating a candidate model for testing).
  - Once validated in simulation or paper trading, the alias will be programmatically or manually updated to `:production` in the registry.

#### B. Full Backtest Table & Plotly Chart Logging
- **[training_evaluator.py](file:///d:/AI-Project/Final/ROBOCHILD/src/analysis/training_evaluator.py)**:
  - Convert backtest execution logs into a `wandb.Table` containing the complete list of trades (Entry, Exit, Action type, execution price, slippage, net fees, and PnL).
  - Log Plotly interactive charts: Equity Curve, Drawdown Progression, and Action Distribution Histogram.

#### C. Dedicated Sweep Configuration Structure
- **[NEW] [sol_sweep.yaml](file:///d:/AI-Project/Final/ROBOCHILD/sweeps/sol_sweep.yaml)**:
  - Position inside a dedicated `sweeps/` subdirectory.
  - Define the Bayesian search metric goals:
    - Primary Metric: `train/explained_variance` (maximize)
    - Secondary Metric: `eval/mean_reward` (maximize)

#### D. Automated Reports
- **[training_evaluator.py](file:///d:/AI-Project/Final/ROBOCHILD/src/analysis/training_evaluator.py)**:
  - Generate an automated WandB Report after important training cycles or sweeps named `SOL Training Report - v{date}` to compare the new candidate run with preceding baseline configurations.

---

### 2. Medium Priority

#### A. Offline & Cache Expiration Management
- **[model_loader.py](file:///d:/AI-Project/Final/ROBOCHILD/src/core/rl_shared/model_loader.py)**:
  - Implement cache expiration logic (e.g., 24 hours).
  - When loading remote model artifacts, check the timestamp of files in `models/wandb_cache/`. If they are older than 24 hours, perform an active check to see if a newer `:production` or `:candidate` model is available on the WandB Registry.
  - Maintain a local file fallback if the network is offline or WandB authentication fails.

#### B. Smart Alerts & Monitoring
- **[trainer.py](file:///d:/AI-Project/Final/ROBOCHILD/src/agent/trainer.py)**:
  - Trigger `wandb.alert()` notifications if:
    - `explained_variance` falls below `0.4` after `50,000` steps.
    - `mean_reward` remains negative for over `100,000` steps.
    - `kl_divergence` exceeds `0.05` indicating policy training instability.

---

## 📂 Proposed File Changes

### [Component: sweeps/]
#### [NEW] [sol_sweep.yaml](file:///d:/AI-Project/Final/ROBOCHILD/sweeps/sol_sweep.yaml)
YAML defining the hyperparameter grids, targeting explained variance and mean reward optimization.

---

### [Component: src/agent/]
#### [MODIFY] [trainer.py](file:///d:/AI-Project/Final/ROBOCHILD/src/agent/trainer.py)
Update callbacks to support smart alerts and package models into artifacts with version tags/aliases.

---

### [Component: src/analysis/]
#### [MODIFY] [training_evaluator.py](file:///d:/AI-Project/Final/ROBOCHILD/src/analysis/training_evaluator.py)
Support log tables, Plotly charts, dynamic project resolving, and automated run reports.

---

### [Component: src/core/rl_shared/]
#### [MODIFY] [model_loader.py](file:///d:/AI-Project/Final/ROBOCHILD/src/core/rl_shared/model_loader.py)
Support WandB registry loading, caching, 24-hour expiration validation, and local fallbacks.

---

## 🧪 Verification Plan

### Automated Tests
- Validate code updates locally with `python validate_pipeline.py`.

### Manual Verification
- Launch a test sweep run and confirm that:
  - The metrics `train/explained_variance` and `eval/mean_reward` are actively tracked.
  - Artifact registry registers versioned files with `:candidate` tags.
  - The model cache on local storage expires and checks for remote updates after 24 hours.
