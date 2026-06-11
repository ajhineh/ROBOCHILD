# 🏛️ Marcos López de Prado Integration Roadmap (Phase 2 & 3)

This roadmap outlines the hybrid implementation plan to transition **ROBOCHILD** from an experimental Reinforcement Learning system to an institutional-grade algorithmic trading system. It integrates the findings from de Prado's paper *"The 7 Reasons Most Machine Learning Funds Fail"* and the `boyboi86/AFML` reference repository.

---

## 📊 Pitfalls Analysis & ROBOCHILD Alignment Matrix

| Pitfall (de Prado) | Current ROBOCHILD Status | Severity | Actionable Resolution |
| :--- | :--- | :--- | :--- |
| **1. Sisyphus Paradigm** | Unified Research & Development teams. | Low | Maintain collaborative, modular agent design. |
| **2. Integer Differentiation** | Using raw returns (integer difference), destroying historical memory. | **High** | Implement **Fractional Differentiation** on prices and Open Interest. |
| **3. Inefficient Sampling** | Implemented static Dollar Bars ($50k). | **High** | Upgrade to **Volatility-Adjusted Dollar Bars** (Completed in Phase 1). |
| **4. Wrong Labeling** | Replaced step-by-step rewards with Triple-Barrier Method. | **Critical** | Refine and enforce **Triple-Barrier Reward** in Gym Environment. |
| **5. Non-IID Weighting** | Standard sample sequence weighting. | Medium | Add Sample uniqueness weights in buffer preprocessing. |
| **6. CV Leakage** | Standard validation sequences without purging. | **High** | Implement **Purged & Embargoed Cross-Validation** in Trainer. |
| **7. Backtest Overfitting** | Local evaluator checking overfitting metrics. | **High** | Restrict validation checks and monitor explained variance. |

---

## 🛠️ Hybrid Implementation Plan

### Task 1: Fractional Differentiation (`frac_diff.py`) — Pitfall 2
*   **Goal**: Establish stationarity on pricing and volume features without destroying long-term memory.
*   **Implementation**:
    1.  Create `src/analysis/frac_diff.py` to calculate fractional differences of order $d$ using a rolling window expansion of weights:
        $$w_k = -w_{k-1} \frac{d - k + 1}{k}$$
    2.  Find the minimum $d \in (0, 1)$ that achieves stationarity using the Augmented Dickey-Fuller (ADF) test (typically $d \approx 0.35$).
    3.  Apply this fractional differentiation to `mid_price`, `spread`, and `basis` in the preprocessing pipeline.

### Task 2: Triple-Barrier Gym Reward (`trading_env.py`) — Pitfall 4
*   **Goal**: Force the RL agents (PPO, SAC, TD3) to learn optimal path-dependent strategies.
*   **Implementation**:
    *   Set vertical barrier (timeout) to $48$ steps.
    *   Calculate TP/SL barriers at trade entry.
    *   Sparsify rewards: assign a reward of $0.0$ for holding periods, and only award positive/negative values when a barrier is hit.

### Task 3: Purged & Embargoed Validation (`trainer.py`) — Pitfall 6
*   **Goal**: Prevent data leakage during model training.
*   **Implementation**:
    1.  Develop a custom validation callback `PurgedValidationCallback` in `trainer.py`.
    2.  Purge validation intervals: remove training steps that overlap in time with any validation sequence.
    3.  Embargo validation intervals: remove $H$ steps immediately following the validation set to eliminate auto-regressive memory leak.

### Task 4: Meta-Ensemble Bet Sizing (`pure_ppo_strategy.py`) — Pitfall 5
*   **Goal**: Scale bet sizes based on ensemble confidence.
*   **Implementation**:
    *   Calculate prediction variance $\sigma^2_{\text{ensemble}}$ between PPO, SAC, and TD3.
    *   Size multiplier:
        $$S = \text{clip}\left(1.0 - \beta \times \sigma^2_{\text{ensemble}}, 0.0, 1.0\right)$$

---

## 🚀 Execution & Training Guidelines (Resolving the 100k Step Failure)

To ensure the newly updated Triple-Barrier Gym Environment converges successfully:
1.  **Increase Training Steps**: Step limits must be raised from 100,000 to **1.5 to 3.0 Million Steps**. Sparse rewards take longer to propagate back.
2.  **Increase Historical Training Window**: Expand the data fetch window from 5 days to **45 days** to capture diverse market regimes.
3.  **Monitor Explained Variance**: Ensure `explained_variance` converges to positive values ($> 0.3$) before using the model in live trading.
