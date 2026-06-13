# WandB Model Versioning & Dynamic Evaluation Integration Plan

This plan details the implementation of WandB Artifact logging (Model Versioning), remote model loading in the live bot, dynamic project routing in the evaluator, and validation environment best practices.

## User Review Required

> [!IMPORTANT]
> - **WandB Online Status**: The live trading bot will require an active internet connection to authenticate with WandB and check for remote artifact updates on startup. We will implement a robust local fallback so that if WandB is down or offline, the bot continues to load the local model files without crashing.
> - **Artifact Version Tags**: The bot will load model artifacts using the `:latest` tag by default. We can also pin it to specific tags (e.g. `:prod` or `:v2`) if we want to manually control which version gets promoted to live trading.

---

## Proposed Changes

### 1. Model Checkpoint Versioning (Training Pipeline)

#### [MODIFY] [trainer.py](file:///d:/AI-Project/Final/ROBOCHILD/src/agent/trainer.py)
- Integrate WandB Artifact creation at the end of the training execution inside `train_agent`.
- After saving the best/final models (`.zip`) and normalization statistics (`.pkl`), package them into a `wandb.Artifact` (type: `"model"`) and upload them to WandB.
- Save the artifact with tags matching the algorithm name (e.g. `ppo`, `sac`, `td3`) and the symbol name.

### 2. Remote Model Loading (Trading Bot)

#### [MODIFY] [model_loader.py](file:///d:/AI-Project/Final/ROBOCHILD/src/core/rl_shared/model_loader.py)
- Integrate remote model retrieval in `load_ppo_model` and `load_ensemble_models`.
- If `USE_WANDB` is enabled in `.env`, attempt to download the target model and normalization pickle from the WandB Artifact registry.
- Store downloaded files in a local cache directory (`models/wandb_cache/`) and load them.
- Implement a try-except fallback mechanism: if the download fails or there is no network connection, fall back to loading the local files from the `models/` directory directly.

### 3. Dynamic Project Routing in Evaluator

#### [MODIFY] [training_evaluator.py](file:///d:/AI-Project/Final/ROBOCHILD/src/analysis/training_evaluator.py)
- Replace hardcoded project pathing `project=f"robochild-{self.symbol}"` with `os.getenv("WANDB_PROJECT", f"robochild-{self.symbol}")`.
- This ensures that when the live bot or training script triggers the evaluator, all backtest metrics and charts are routed to the current active project (`ROBOCHILD-SOL`) rather than cluttering old or separate projects.

---

## Verification Plan

### Automated Tests
- Run `python validate_pipeline.py` locally to ensure there are no syntax or namespace errors introduced in the updated modules.

### Manual Verification
1. **Mock Artifact Log**: Run a short, dummy 1,000-step training locally to verify that weights and normalization pickles are successfully uploaded to WandB as a versioned artifact.
2. **Mock Artifact Download**: Execute a model loader test script locally to verify that it downloads the target artifact from WandB, places it in the cache, and loads it successfully.
3. **Offline Fallback**: Disable internet connection/WandB auth and verify that the loader falls back gracefully to loading local models without raising exceptions.
