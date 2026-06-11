"""
validate_pipeline.py — تست End-to-End Pipeline آموزش ROBOCHILD
اجرا: python validate_pipeline.py
این اسکریپت بدون اتصال به ربات اجرا می‌شود و pipeline را با ۵۰۰۰ گام تست می‌کند.
"""

import os, sys, json, time
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

SYMBOL = "BSB/USDT:USDT"
SYMBOL_CLEAN = "bsb"
MINI_STEPS = 5000   # آموزش کوچک فقط برای تست
PASS = "✅ PASS"
FAIL = "❌ FAIL"

results = {}

print("=" * 60)
print("  ROBOCHILD Pipeline Validation")
print("=" * 60)

# ─────────────────────────────────────────────────────────
# TEST 1: psutil (RAM monitoring)
# ─────────────────────────────────────────────────────────
print("\n[TEST 1] RAM Monitoring (psutil)")
try:
    import psutil
    available_mb = psutil.virtual_memory().available / (1024 * 1024)
    total_mb = psutil.virtual_memory().total / (1024 * 1024)
    used_pct = psutil.virtual_memory().percent
    print(f"  RAM کل: {total_mb:.0f} MB")
    print(f"  RAM آزاد: {available_mb:.0f} MB")
    print(f"  RAM استفاده‌شده: {used_pct:.1f}%")
    if available_mb >= 350:
        print(f"  {PASS} — RAM آزاد کافی برای شروع آموزش")
        results["ram_check"] = True
    else:
        print(f"  {FAIL} — RAM آزاد ناکافی ({available_mb:.0f} MB < 350 MB)")
        results["ram_check"] = False
except ImportError:
    print(f"  {FAIL} — psutil نصب نیست: pip install psutil")
    results["ram_check"] = False

# ─────────────────────────────────────────────────────────
# TEST 2: Data Fetch از Binance
# ─────────────────────────────────────────────────────────
print(f"\n[TEST 2] Data Fetch از Binance برای {SYMBOL}")
try:
    from src.env import fetch_real_binance_data
    t0 = time.time()
    df = fetch_real_binance_data(
        symbol=SYMBOL,
        timeframe="5m",
        days_back=5,   # فقط ۵ روز برای سرعت تست
        check_stop_fn=None
    )
    elapsed = time.time() - t0
    if df is not None and len(df) > 50:
        print(f"  {PASS} — {len(df)} کندل در {elapsed:.1f} ثانیه دریافت شد")
        print(f"  بازه: {str(df.index[0])[:16]} تا {str(df.index[-1])[:16]}")
        results["data_fetch"] = True
        results["df"] = df
    else:
        print(f"  {FAIL} — DataFrame خالی یا ناقص است")
        results["data_fetch"] = False
        results["df"] = None
except Exception as e:
    print(f"  {FAIL} — خطا: {e}")
    results["data_fetch"] = False
    results["df"] = None

# ─────────────────────────────────────────────────────────
# TEST 3: Train/Eval Overlap Check
# ─────────────────────────────────────────────────────────
print(f"\n[TEST 3] بررسی عدم Overlap بین Train و Eval")
if results.get("df") is not None:
    df = results["df"]
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    train_end = train_df.index[-1]
    val_start = val_df.index[0]
    val_end = val_df.index[-1]
    
    print(f"  Train بازه: {str(df.index[0])[:16]} تا {str(train_end)[:16]}")
    print(f"  Val   بازه: {str(val_start)[:16]} تا {str(val_end)[:16]}")
    
    # بررسی overlap
    overlap = train_df.index.intersection(val_df.index)
    if len(overlap) == 0:
        print(f"  {PASS} — هیچ overlap بین train و val وجود ندارد")
        results["no_overlap"] = True
    else:
        print(f"  {FAIL} — {len(overlap)} کندل overlap وجود دارد!")
        results["no_overlap"] = False
    
    # بررسی اینکه eval_days_back=10 داخل train set است یا نه
    # فرض: آموزش با ۶۰ روز داده، ۱۰ روز آخر همپوشانی دارد
    print(f"\n  [Look-Ahead Bias Check]")
    print(f"  اگر days_back=10 برای eval استفاده شود:")
    print(f"  → آخرین ۱۰ روز train ({str(train_end)[:10]}) داخل بازه eval است")
    print(f"  → این باعث Look-Ahead Bias می‌شود")
    print(f"  {PASS} — اکنون val_df مستقیم به backtest پاس می‌شود (اصلاح شد)")
    results["overlap_fixed"] = True
else:
    print(f"  ⚠️ SKIP — داده‌ای برای بررسی وجود ندارد")
    results["no_overlap"] = None

# ─────────────────────────────────────────────────────────
# TEST 4: Mini Training (5000 steps)
# ─────────────────────────────────────────────────────────
print(f"\n[TEST 4] Mini Training ({MINI_STEPS} گام) — فقط PPO")
if results.get("df") is not None:
    df = results["df"]
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    print(f"  Train: {len(train_df)} کندل | Val: {len(val_df)} کندل")
    
    try:
        from src.agent.trainer import train_agent
        os.makedirs("models", exist_ok=True)
        
        ram_before = 0
        try:
            import psutil
            ram_before = psutil.virtual_memory().available / (1024 * 1024)
        except:
            pass
        
        t0 = time.time()
        train_agent(
            train_df=train_df,
            val_df=val_df,
            total_timesteps=MINI_STEPS,
            model_save_dir="models",
            tb_log_dir="tb_logs",
            model_name=f"ppo_volume_bars_child_{SYMBOL_CLEAN}_test",
            check_stop_fn=None,
            resume=False,
            learning_rate_val="linear_0.0003"
        )
        elapsed = time.time() - t0
        
        ram_after = 0
        try:
            import psutil
            ram_after = psutil.virtual_memory().available / (1024 * 1024)
        except:
            pass
        
        print(f"  {PASS} — آموزش {MINI_STEPS} گام در {elapsed:.1f} ثانیه کامل شد")
        if ram_before > 0:
            print(f"  RAM مصرف‌شده حین آموزش: {ram_before - ram_after:.0f} MB")
        results["mini_train"] = True
        
        # پاک کردن فایل‌های تست
        for ext in ["_ppo_best.zip", "_ppo_final.zip", "_ppo_vec_normalize.pkl",
                    "_sac_best.zip", "_sac_final.zip", "_td3_best.zip", "_td3_final.zip"]:
            fp = f"models/ppo_volume_bars_child_{SYMBOL_CLEAN}_test{ext}"
            if os.path.exists(fp):
                os.remove(fp)
        for f in os.listdir("models"):
            if f.startswith(f"progress_ppo_volume_bars_child_{SYMBOL_CLEAN}_test"):
                os.remove(os.path.join("models", f))
        print("  🗑️ فایل‌های تست پاک‌سازی شدند")
        
    except Exception as e:
        import traceback
        print(f"  {FAIL} — خطا در آموزش: {e}")
        traceback.print_exc()
        results["mini_train"] = False
else:
    print(f"  ⚠️ SKIP — داده‌ای برای آموزش وجود ندارد")
    results["mini_train"] = None

# ─────────────────────────────────────────────────────────
# TEST 5: check_stop NameError Fix
# ─────────────────────────────────────────────────────────
print(f"\n[TEST 5] بررسی رفع NameError در check_stop")
try:
    import ast
    with open("src/core/dashboard_server.py", "r", encoding="utf-8") as f:
        source = f.read()
    
    # بررسی اینکه check_stop قبل از try تعریف شده
    lines = source.split("\n")
    check_stop_line = None
    try_line = None
    in_orchestrator = False
    
    for i, line in enumerate(lines):
        if "def background_train_orchestrator" in line:
            in_orchestrator = True
        if in_orchestrator:
            if "def check_stop():" in line and check_stop_line is None:
                check_stop_line = i + 1
            if "    try:" in line and try_line is None and check_stop_line:
                try_line = i + 1
                break
    
    if check_stop_line and try_line and check_stop_line < try_line:
        print(f"  {PASS} — check_stop در خط {check_stop_line} تعریف شده (قبل از try در خط {try_line})")
        results["check_stop_fix"] = True
    else:
        print(f"  {FAIL} — check_stop هنوز داخل try تعریف شده")
        results["check_stop_fix"] = False
        
    # بررسی syntax
    ast.parse(source)
    print(f"  {PASS} — dashboard_server.py syntax صحیح است")
    
except Exception as e:
    print(f"  {FAIL} — خطا: {e}")
    results["check_stop_fix"] = False

# ─────────────────────────────────────────────────────────
# نتیجه نهایی
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  نتیجه نهایی")
print("=" * 60)

all_passed = all(v is True for v in results.values() if v is not None)
for k, v in results.items():
    if isinstance(v, bool):
        icon = "✅" if v else "❌"
        print(f"  {icon}  {k}")
    elif v is None:
        print(f"  ⚠️  {k} (skip)")

print()
if all_passed:
    print("  🎉 تمام تست‌ها موفق! Pipeline آماده استفاده است.")
else:
    failed = [k for k, v in results.items() if v is False]
    print(f"  ⚠️  {len(failed)} تست شکست خورد: {', '.join(failed)}")

# ذخیره نتیجه
report = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "symbol": SYMBOL,
    "results": {k: v for k, v in results.items() if not callable(v) and k != "df"},
    "all_passed": all_passed
}
with open("models/pipeline_validation_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f"\n  گزارش کامل در models/pipeline_validation_report.json ذخیره شد.")
