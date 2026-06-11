import os
import re
import json
import time
import logging
from collections import deque
import asyncio

logger = logging.getLogger("ROBORDER.Dashboard.API")

def handle_api_status(handler, global_engine, global_executor, scan_existing_models, global_pings, Config):
    """ارائه وضعیت لحظه‌ای متغیرها، پوزیشن‌های باز، آمار دروداون و اندیکاتورهای ربات"""
    portfolio = {
        "balance": Config.CURRENT_BALANCE,
        "unrealized_pnl": 0.0,
        "equity": Config.CURRENT_BALANCE,
        "drawdown": 0.0,
        "status": "Paper Simulation"
    }
    
    open_positions = []

    if global_executor:
        portfolio["drawdown"] = round(global_executor.current_drawdown, 2)
        portfolio["status"] = "Live Futures" if global_executor.live_trading else "Paper Simulation"
        
        pnl_sum = 0.0
        used_margin = 0.0
        for sym, pos in global_executor.open_positions.items():
            pos_pnl = 0.0
            if global_engine and sym in global_engine.latest_lob_results:
                mid = global_engine.latest_lob_results[sym]["mid_price"]
                entry = pos["entry_price"]
                if pos["side"] == "long":
                    pos_pnl = ((mid - entry) / entry) * 100.0 * pos["leverage"]
                else:
                    pos_pnl = ((entry - mid) / entry) * 100.0 * pos["leverage"]
                
            pnl_sum += (pos.get("amount", 0.0) * (pos_pnl / 100.0))
            used_margin += pos.get("amount", 0.0)
            
            open_positions.append({
                "symbol": sym,
                "side": pos["side"],
                "leverage": pos["leverage"],
                "entry_price": pos["entry_price"],
                "tp": pos["tp"],
                "sl": pos["sl"],
                "opened_at": pos.get("opened_at", time.time()),
                "max_holding_seconds": 14400,
                "pnl": round(pos_pnl, 2)
            })
        
        portfolio["unrealized_pnl"] = round(pnl_sum, 2)
        portfolio["equity"] = round(Config.CURRENT_BALANCE + pnl_sum, 2)
        portfolio["balance"] = round(Config.CURRENT_BALANCE - used_margin, 2)

    active_bots = []
    if global_engine:
        for sym in Config.SYMBOLS:
            lob = global_engine.latest_lob_results.get(sym)
            active = global_engine.yoyo.active_trades.get(sym)
            
            dex_trades = global_engine.recent_dex_trades.get(sym, [])
            now_ms_dash = int(time.time() * 1000)
            dex_trades_10s = [t for t in dex_trades if (now_ms_dash - t["timestamp"]) <= 10000]
            dex_buy = sum([t["amount"] for t in dex_trades_10s if t["side"] == "buy"])
            dex_sell = sum([t["amount"] for t in dex_trades_10s if t["side"] == "sell"])

            symbol_clean = sym.split('/')[0].lower()
            
            tp_bps = Config.TAKE_PROFIT_BPS
            sl_bps = Config.STOP_LOSS_BPS
            config_path_sym = os.path.join("models", f"config_{symbol_clean}.json")
            if os.path.exists(config_path_sym):
                try:
                    with open(config_path_sym, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                        tp_bps = cfg.get("take_profit_bps", tp_bps)
                        sl_bps = cfg.get("stop_loss_bps", sl_bps)
                except Exception:
                    pass

            has_ppo = os.path.exists(f"models/ppo_volume_bars_child_{symbol_clean}_ppo_final.zip") or os.path.exists(f"models/ppo_volume_bars_child_{symbol_clean}_ppo_best.zip")
            has_sac = os.path.exists(f"models/ppo_volume_bars_child_{symbol_clean}_sac_final.zip") or os.path.exists(f"models/ppo_volume_bars_child_{symbol_clean}_sac_best.zip")
            has_td3 = os.path.exists(f"models/ppo_volume_bars_child_{symbol_clean}_td3_final.zip") or os.path.exists(f"models/ppo_volume_bars_child_{symbol_clean}_td3_best.zip")
            has_old_model = os.path.exists(f"models/ppo_volume_bars_child_{symbol_clean}_final.zip") or os.path.exists(f"models/ppo_volume_bars_child_{symbol_clean}_best.zip")
            
            has_model = has_ppo or has_old_model
            
            model_age_days = -1
            needs_retrain = False
            retrain_reason = ""
            
            model_paths_to_check = [
                f"models/ppo_volume_bars_child_{symbol_clean}_ppo_final.zip",
                f"models/ppo_volume_bars_child_{symbol_clean}_ppo_best.zip",
                f"models/ppo_volume_bars_child_{symbol_clean}_final.zip",
                f"models/ppo_volume_bars_child_{symbol_clean}_best.zip"
            ]
            
            trained_time = 0
            for path in model_paths_to_check:
                if os.path.exists(path):
                    mtime = os.path.getmtime(path)
                    if mtime > trained_time:
                        trained_time = mtime
                        
            if trained_time > 0:
                model_age_days = int((time.time() - trained_time) / (24 * 3600))
                if model_age_days >= 30:
                    needs_retrain = True
                    retrain_reason = f"مدل {model_age_days} روز پیش آموزش دیده است. بر اساس منطق Rolling Window، آموزش مجدد توصیه می‌شود."

            bot_data = {
                "symbol": sym,
                "status": "Flat" if not active else f"Active {active['side'].upper()} ({active['status'].upper()})",
                "raw_obi": 0.0,
                "market_buy_vol": 0.0,
                "market_sell_vol": 0.0,
                "dex_buy_vol": round(dex_buy, 2),
                "dex_sell_vol": round(dex_sell, 2),
                "take_profit_bps": tp_bps,
                "stop_loss_bps": sl_bps,
                "spoof_type": "none",
                "mid_price": 0.0,
                "has_model": has_model,
                "has_ppo": has_ppo,
                "has_sac": has_sac,
                "has_td3": has_td3,
                "has_old_model": has_old_model,
                "model_age_days": model_age_days,
                "needs_retrain": needs_retrain,
                "retrain_reason": retrain_reason,
                "leverage": active["leverage"] if active else 15
            }

            if lob:
                bot_data.update({
                    "raw_obi": round(lob["raw_obi"], 2),
                    "market_buy_vol": round(lob["market_buy_vol"], 1),
                    "market_sell_vol": round(lob["market_sell_vol"], 1),
                    "spoof_type": lob["spoof_type"],
                    "mid_price": round(lob["mid_price"], 6)
                })
            
            active_bots.append(bot_data)

    available_models = scan_existing_models()

    response_data = {
        "active_settings": {
            "EXCHANGE_ID": Config.EXCHANGE_ID,
            "ROBORDER_LIVE": Config.ROBORDER_LIVE,
            "QUOTE_DENOMINATION": Config.QUOTE_DENOMINATION,
            "CUSTOM_WS_ENDPOINT": Config.CUSTOM_WS_ENDPOINT,
            "HELIUS_WS_URL": Config.HELIUS_WS_URL,
            "QUICKNODE_WS_URL": Config.QUICKNODE_WS_URL,
            "LOB_DEPTH_LEVELS": Config.LOB_DEPTH_LEVELS,
            "TRADE_WINDOW_SECONDS": Config.TRADE_WINDOW_SECONDS,
            "SPOOF_THRESHOLD_PCT": Config.SPOOF_THRESHOLD_PCT,
            "MOMENTUM_WINDOW_MS": Config.MOMENTUM_WINDOW_MS,
            "SPREAD_THRESHOLD_BPS": Config.SPREAD_THRESHOLD_BPS,
            "TAKE_PROFIT_BPS": Config.TAKE_PROFIT_BPS,
            "STOP_LOSS_BPS": Config.STOP_LOSS_BPS,
            "COOLDOWN_MS": Config.COOLDOWN_MS,
            "MAX_CONCURRENT_POSITIONS": Config.MAX_CONCURRENT_POSITIONS,
            "MAX_DRAWDOWN_LIMIT_USDT": Config.MAX_DRAWDOWN_LIMIT_USDT,
            "INITIAL_BALANCE": Config.INITIAL_BALANCE,
            "TRADE_CAPITAL_PCT": Config.TRADE_CAPITAL_PCT,
            "USE_ONLY_PPO": Config.USE_ONLY_PPO,
            "USE_YOYO_STRATEGY": Config.USE_YOYO_STRATEGY,
            "YOYO_RISK_PCT": Config.YOYO_RISK_PCT,
            "BYPASSED_FILTERS": Config.BYPASSED_FILTERS,
            "SCREENER_ENABLED": Config.SCREENER_ENABLED
        },
        "available_models": available_models,
        "active_bots": active_bots,
        "open_positions": open_positions,
        "portfolio": portfolio,
        "pings": global_pings
    }
    handler.send_json(response_data)


def handle_api_add_symbol(handler, body, global_engine, global_executor, global_loop, active_trainings, active_analyses, auto_activate_symbol, log_event, background_train_orchestrator, Config, threading):
    """افزودن ارز: راه‌اندازی فرآیند آموزش هوش مصنوعی جدید یا فعال‌سازی مدل از قبل پیاده‌سازی شده"""
    symbol = body.get("symbol", "").upper().strip()
    symbol = re.sub(r'[^a-zA-Z0-9/:-]', '', symbol)
    resume = bool(body.get("resume", False))
    learning_rate = body.get("learning_rate", "linear_0.0003")

    if not symbol:
        handler.send_json({"success": False, "message": "ارز نامشخص است"}, 400)
        return

    symbol_clean = symbol.split('/')[0].lower()

    if Config.USE_YOYO_STRATEGY and not resume:
        # در استراتژی YoYo نیازی به مدل هوش مصنوعی نیست و مستقیماً ترید آغاز می‌شود
        if symbol in Config.SYMBOLS:
            handler.send_json({"success": True, "message": f"جفت ارز {symbol} از قبل فعال و در حال ترید است."})
            return
        
        Config.SYMBOLS.append(symbol)
        from src.config import save_env_values
        save_env_values({"SYMBOLS": ",".join(Config.SYMBOLS)})
        log_event(f"➕ جفت ارز {symbol} اضافه شد و در فایل .env ذخیره گردید.")
        
        if global_engine:
            if symbol not in global_engine.symbols:
                global_engine.symbols.append(symbol)
            if hasattr(global_engine, "yoyo") and global_engine.yoyo:
                if symbol not in global_engine.yoyo.symbols:
                    global_engine.yoyo.symbols.append(symbol)
                    global_engine.yoyo.candles_1m[symbol] = []
                    global_engine.yoyo.candles_3m[symbol] = []
                    global_engine.yoyo.candles_15m[symbol] = []
                    global_engine.yoyo.current_1m[symbol] = None
                    global_engine.yoyo.current_3m[symbol] = None
                    global_engine.yoyo.current_15m[symbol] = None
                    global_engine.yoyo.last_order_placed_time[symbol] = 0.0
                
                import asyncio as _asyncio
                exch = global_executor.exchange if (global_executor and hasattr(global_executor, 'exchange')) else None
                if global_loop and exch:
                    try:
                        _asyncio.run_coroutine_threadsafe(global_engine.yoyo.initialize_candles(exch), global_loop)
                    except Exception as e:
                        log_event(f"⚠️ خطای غیرمنتظره در مقداردهی شمع‌های YoYo: {e}")
                        global_engine.yoyo._generate_mock_historical_candles(symbol)
                else:
                    global_engine.yoyo._generate_mock_historical_candles(symbol)
            
            if symbol not in global_engine.recent_dex_trades:
                global_engine.recent_dex_trades[symbol] = deque()

        handler.send_json({"success": True, "message": f"جفت ارز {symbol} با موفقیت به استراتژی YoYo اضافه شد و شمع‌های تاریخی آن آماده‌سازی گردید."})
        return

    if symbol_clean in active_trainings:
        handler.send_json({
            "success": False,
            "message": f"فرآیند آموزش هوش مصنوعی برای {symbol} در پس‌زمینه در جریان است. لطفاً منتظر بمانید."
        }, 400)
        return

    if active_analyses.get(symbol_clean) == "running":
        handler.send_json({
            "success": False,
            "message": "امکان شروع آموزش در حین فرآیند آنالیزور مدل وجود ندارد. لطفاً تا پایان تحلیل صبر کنید."
        }, 400)
        return

    if not resume:
        # اگر کاربر آموزش جدید و تمیز از صفر درخواست کرده است، فایل‌های مدل قبلی و وضعیت پیشرفت آنها را از هارد پاک می‌کنیم
        files_to_clean = []
        for algo in ["ppo", "sac", "td3"]:
            files_to_clean.append(f"models/ppo_volume_bars_child_{symbol_clean}_{algo}_final.zip")
            files_to_clean.append(f"models/ppo_volume_bars_child_{symbol_clean}_{algo}_best.zip")
            files_to_clean.append(f"models/ppo_volume_bars_child_{symbol_clean}_{algo}_vec_normalize.pkl")
        files_to_clean.extend([
            f"models/ppo_volume_bars_child_{symbol_clean}_final.zip",
            f"models/ppo_volume_bars_child_{symbol_clean}_best.zip",
            f"models/ppo_volume_bars_child_{symbol_clean}_vec_normalize.pkl",
            f"models/progress_ppo_volume_bars_child_{symbol_clean}.json"
        ])
        for f_path in files_to_clean:
            if os.path.exists(f_path):
                try:
                    os.remove(f_path)
                except Exception as e:
                    logger.warning(f"Failed to clear old model file {f_path} on fresh restart: {e}")

    model_file = f"models/ppo_volume_bars_child_{symbol_clean}_ppo_best.zip"
    if not os.path.exists(model_file):
        model_file = f"models/ppo_volume_bars_child_{symbol_clean}_ppo_final.zip"
    if not os.path.exists(model_file):
        model_file = f"models/ppo_volume_bars_child_{symbol_clean}_best.zip"
    if not os.path.exists(model_file):
        model_file = f"models/ppo_volume_bars_child_{symbol_clean}_final.zip"

    # فقط در صورتی که دکمه از سرگیری (resume) روشن باشد، بررسی می‌کنیم مدل از قبل وجود دارد یا خیر تا فعالش کنیم.
    # در غیر این صورت (وقتی resume غیرفعال است)، مستقیماً از ابتدا آموزش را شروع می‌کنیم حتی اگر فایلی مانده باشد.
    if resume and os.path.exists(model_file):
        # اتصال اولیه به سیستم ترید زنده در صورت عدم وجود
        is_already_trading = symbol in Config.SYMBOLS
        if not is_already_trading:
            Config.SYMBOLS.append(symbol)
            from src.config import save_env_values
            save_env_values({"SYMBOLS": ",".join(Config.SYMBOLS)})
            log_event(f"➕ جفت ارز {symbol} اضافه شد و در فایل .env ذخیره گردید.")
            
            if global_engine:
                if symbol not in global_engine.symbols:
                    global_engine.symbols.append(symbol)
                if hasattr(global_engine, "yoyo") and global_engine.yoyo:
                    if symbol not in global_engine.yoyo.symbols:
                        global_engine.yoyo.symbols.append(symbol)
                        global_engine.yoyo.candles_1m[symbol] = []
                        global_engine.yoyo.candles_3m[symbol] = []
                        global_engine.yoyo.candles_15m[symbol] = []
                        global_engine.yoyo.current_1m[symbol] = None
                        global_engine.yoyo.current_3m[symbol] = None
                        global_engine.yoyo.current_15m[symbol] = None
                        global_engine.yoyo.last_order_placed_time[symbol] = 0.0
                        
                        import asyncio as _asyncio
                        exch = global_executor.exchange if (global_executor and hasattr(global_executor, 'exchange')) else None
                        if global_loop and exch:
                            try:
                                _asyncio.run_coroutine_threadsafe(global_engine.yoyo.initialize_candles(exch), global_loop)
                            except Exception as e:
                                log_event(f"⚠️ خطای غیرمنتظره در مقداردهی شمع‌های YoYo: {e}")
                                global_engine.yoyo._generate_mock_historical_candles(symbol)
                        else:
                            global_engine.yoyo._generate_mock_historical_candles(symbol)
                            
                if symbol not in global_engine.recent_dex_trades:
                    global_engine.recent_dex_trades[symbol] = deque()
            
            log_event(f"مدل هوش مصنوعی یافت شد! جفت ارز {symbol} به سیستم ترید زنده متصل شد.")

        # همزمان آموزش را نیز از سر می‌گیریم
        steps = int(body.get("steps", 200000))
        log_event(f"🔄 درخواست از سرگیری (Fine-tune) مدل {symbol} با {steps:,} گام جدید و نرخ یادگیری {learning_rate}...")
        
        thread = threading.Thread(
            target=background_train_orchestrator, 
            args=(symbol, steps), 
            kwargs={"resume": resume, "learning_rate": learning_rate},
            daemon=True
        )
        thread.start()

        msg = f"مدل هوش مصنوعی متصل گردید. همزمان فرآیند ارتقا و ادامه آموزش مدل {symbol} با {steps:,} گام جدید استارت خورد."
        handler.send_json({
            "success": True,
            "message": msg
        })
    else:
        steps = int(body.get("steps", 200000))
        if resume:
            log_event(f"🔄 درخواست از سرگیری (Fine-tune) مدل {symbol} با {steps:,} گام جدید و نرخ یادگیری {learning_rate}...")
        else:
            log_event(f"🔍 مدل شبکه عصبی برای {symbol} یافت نشد یا درخواست ساخت مجدد صادر شده است. تریگر آموزش جدید در پس‌زمینه...")
        
        thread = threading.Thread(
            target=background_train_orchestrator, 
            args=(symbol, steps), 
            kwargs={"resume": resume, "learning_rate": learning_rate},
            daemon=True
        )
        thread.start()

        msg = f"فرآیند آموزش شبکه عصبی با بودجه {steps:,} گام استارت خورد."
        if resume:
            msg = f"فرآیند از سرگیری و ارتقای مدل {symbol} با {steps:,} گام جدید استارت خورد."

        handler.send_json({
            "success": True,
            "message": msg
        })


def handle_api_remove_symbol(handler, body, global_engine, global_executor, global_loop, active_trainings, training_stops, log_event, Config):
    """حذف ارز: متوقف کردن آموزش هوش مصنوعی یا حذف جفت ارز از لیست ترید فعال"""
    symbol = body.get("symbol", "").upper().strip()
    symbol = re.sub(r'[^a-zA-Z0-9/:-]', '', symbol)
    delete_model = bool(body.get("delete_model", False))
    
    symbol_clean = symbol.split('/')[0].lower()
    deleted_files = []

    if delete_model:
        files_to_delete = []
        for algo in ["ppo", "sac", "td3"]:
            files_to_delete.append(f"models/ppo_volume_bars_child_{symbol_clean}_{algo}_final.zip")
            files_to_delete.append(f"models/ppo_volume_bars_child_{symbol_clean}_{algo}_best.zip")
            files_to_delete.append(f"models/ppo_volume_bars_child_{symbol_clean}_{algo}_vec_normalize.pkl")
        files_to_delete.extend([
            f"models/ppo_volume_bars_child_{symbol_clean}_final.zip",
            f"models/ppo_volume_bars_child_{symbol_clean}_best.zip",
            f"models/ppo_volume_bars_child_{symbol_clean}_vec_normalize.pkl",
            f"models/progress_ppo_volume_bars_child_{symbol_clean}.json"
        ])
        
        for file_path in files_to_delete:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    deleted_files.append(os.path.basename(file_path))
                except Exception as e:
                    log_event(f"⚠️ خطا در حذف فایل {file_path}: {e}")

    if symbol_clean in active_trainings:
        training_stops[symbol_clean] = True
        log_event(f"🛑 دستور لغو آموزش هوش مصنوعی برای {symbol} ({symbol_clean}) صادر شد.")
        msg = f"آموزش شبکه عصبی برای {symbol} متوقف شد."
        if deleted_files:
            msg += f" فایل‌های مدل نیز پاک‌سازی شدند: {', '.join(deleted_files)}"
        handler.send_json({"success": True, "message": msg})
        return

    progress_path = os.path.join("models", f"progress_ppo_volume_bars_child_{symbol_clean}.json")
    if os.path.exists(progress_path):
        try:
            with open(progress_path, "r", encoding="utf-8") as _pf:
                _pdata = json.load(_pf)
            if _pdata.get("status") in ("training", "error"):
                os.remove(progress_path)
                log_event(f"🗑️ فایل وضعیت آموزش قدیمی برای {symbol} پاک‌سازی شد.")
        except Exception:
            pass

    target_symbol = None
    clean_symbol = symbol.split(":")[0].upper().strip()
    for sym in Config.SYMBOLS:
        if sym.upper().strip() == symbol or sym.split(":")[0].upper().strip() == clean_symbol:
            target_symbol = sym
            break

    if target_symbol:
        Config.SYMBOLS.remove(target_symbol)
        from src.config import save_env_values
        success_save = save_env_values({"SYMBOLS": ",".join(Config.SYMBOLS)})
        if success_save:
            log_event(f"➖ جفت ارز {target_symbol} از لیست ترید فعال حذف و تنظیمات .env بروز شد.")
        else:
            log_event(f"⚠️ جفت ارز {target_symbol} در حافظه موقت حذف شد ولی نوشتن در .env خطا داشت.")
        
        if global_engine:
            if target_symbol in global_engine.symbols:
                global_engine.symbols.remove(target_symbol)
            if hasattr(global_engine, "yoyo") and global_engine.yoyo:
                if target_symbol in global_engine.yoyo.symbols:
                    global_engine.yoyo.symbols.remove(target_symbol)
                global_engine.yoyo.candles_1m.pop(target_symbol, None)
                global_engine.yoyo.candles_3m.pop(target_symbol, None)
                global_engine.yoyo.candles_15m.pop(target_symbol, None)
                global_engine.yoyo.current_1m.pop(target_symbol, None)
                global_engine.yoyo.current_3m.pop(target_symbol, None)
                global_engine.yoyo.current_15m.pop(target_symbol, None)
                global_engine.yoyo.active_trades.pop(target_symbol, None)
                global_engine.yoyo.last_order_placed_time.pop(target_symbol, None)
        
        if global_executor and target_symbol in global_executor.open_positions:
            try:
                import asyncio as _asyncio
                if global_loop:
                    _asyncio.run_coroutine_threadsafe(
                        global_executor.execute_exit(target_symbol, 0.0, "FORCE_DASHBOARD_REMOVE"),
                        global_loop
                    )
                log_event(f"🚪 پوزیشن باز جفت ارز {target_symbol} با موفقیت در صرافی بسته شد.")
            except Exception as e:
                log_event(f"⚠️ خطا در بستن پوزیشن {target_symbol}: {e}")

        msg = f"جفت ارز {target_symbol} با موفقیت از سیستم ترید زنده حذف شد."
        if deleted_files:
            msg += f" فایل‌های شبکه عصبی نیز حذف شدند: {', '.join(deleted_files)}"
        handler.send_json({"success": True, "message": msg})
    else:
        handler.send_json({"success": False, "message": "ارز مدنظر در لیست فعال یافت نشد."}, 404)


def handle_api_close_position(handler, body, global_engine, global_executor, global_loop, log_event, Config):
    """بستن فوری پوزیشن یک ارز بدون حذف آن از لیست نمادها"""
    symbol = body.get("symbol", "").upper().strip()
    symbol = re.sub(r'[^a-zA-Z0-9/:-]', '', symbol)
    if not symbol:
        handler.send_json({"success": False, "message": "ارز نامشخص است"}, 400)
        return

    target_symbol = None
    clean_symbol = symbol.split(":")[0].upper().strip()
    for sym in Config.SYMBOLS:
        if sym.upper().strip() == symbol or sym.split(":")[0].upper().strip() == clean_symbol:
            target_symbol = sym
            break

    if not target_symbol:
        handler.send_json({"success": False, "message": "ارز مدنظر در لیست فعال یافت نشد."}, 404)
        return

    log_event(f"🚪 درخواست بستن فوری پوزیشن برای {target_symbol} دریافت شد.")

    closed_locally = False
    import asyncio as _asyncio

    if global_engine and hasattr(global_engine, "yoyo") and global_engine.yoyo:
        yoyo = global_engine.yoyo
        if target_symbol in yoyo.active_trades:
            current_price = 0.0
            if target_symbol in global_engine.latest_lob_results:
                current_price = global_engine.latest_lob_results[target_symbol]["mid_price"]
            else:
                trade = yoyo.active_trades[target_symbol]
                current_price = trade.get("entry_price", 0.0)

            now_ms = int(time.time() * 1000)
            try:
                if hasattr(yoyo, "force_close_position"):
                    closed_locally = yoyo.force_close_position(target_symbol, current_price, now_ms)
                else:
                    trade = yoyo.active_trades[target_symbol]
                    yoyo._close_ppo_position(target_symbol, trade, current_price, 0.0, "FORCE_DASHBOARD_CLOSE", now_ms)
                    closed_locally = True
            except Exception as e:
                log_event(f"⚠️ خطا در بستن پوزیشن استراتژی: {e}")

    if not closed_locally and global_executor and target_symbol in global_executor.open_positions:
        try:
            pos = global_executor.open_positions[target_symbol]
            pos_pnl = 0.0
            pnl_usdt = 0.0
            if global_engine and target_symbol in global_engine.latest_lob_results:
                mid = global_engine.latest_lob_results[target_symbol]["mid_price"]
                entry = pos["entry_price"]
                if pos["side"] == "long":
                    pos_pnl = ((mid - entry) / entry) * 100.0 * pos["leverage"]
                else:
                    pos_pnl = ((entry - mid) / entry) * 100.0 * pos["leverage"]
                pnl_usdt = pos.get("amount", 0.0) * (pos_pnl / 100.0)
            
            if global_loop:
                _asyncio.run_coroutine_threadsafe(
                    global_executor.execute_exit(target_symbol, pnl_usdt, "FORCE_DASHBOARD_CLOSE"),
                    global_loop
                )
            closed_locally = True
            log_event(f"🚪 پوزیشن باز {target_symbol} مستقیماً در صرافی بسته شد.")
        except Exception as e:
            log_event(f"⚠️ خطا در بستن مستقیم پوزیشن صرافی {target_symbol}: {e}")

    if closed_locally:
        handler.send_json({"success": True, "message": f"پوزیشن باز جفت ارز {target_symbol} با موفقیت در صرافی بسته شد."})
    else:
        handler.send_json({"success": False, "message": "هیچ پوزیشن باز یا معامله فعالی برای این جفت ارز یافت نشد."}, 400)


def handle_api_set_settings(handler, body, global_engine, global_executor, global_loop, log_event, Config):
    """ذخیره تنظیمات عددی جدید ارسال شده از مرورگر مستقیماً درون فایل متغیرهای محیطی .env و اعمال آنی به موتورها"""
    updates = {}
    valid_keys = [
        "ROBORDER_LIVE", "EXCHANGE_ID", "QUOTE_DENOMINATION", "CUSTOM_WS_ENDPOINT",
        "HELIUS_WS_URL", "QUICKNODE_WS_URL", "LOB_DEPTH_LEVELS", "TRADE_WINDOW_SECONDS",
        "SPOOF_THRESHOLD_PCT", "MOMENTUM_WINDOW_MS", "SPREAD_THRESHOLD_BPS",
        "TAKE_PROFIT_BPS", "STOP_LOSS_BPS", "COOLDOWN_MS", "MAX_CONCURRENT_POSITIONS",
        "MAX_DRAWDOWN_LIMIT_USDT", "INITIAL_BALANCE", "TRADE_CAPITAL_PCT", "USE_ONLY_PPO",
        "USE_YOYO_STRATEGY", "YOYO_RISK_PCT", "BYPASSED_FILTERS", "SCREENER_ENABLED"
    ]

    for key, val in body.items():
        if key in valid_keys:
            if isinstance(val, bool):
                updates[key] = "true" if val else "false"
            else:
                updates[key] = str(val).strip()

    from src.config import save_env_values
    success = save_env_values(updates)
    if success:
        Config.reload()

        if global_executor:
            global_executor.live_trading = Config.ROBORDER_LIVE
            global_executor.max_concurrent_positions = Config.MAX_CONCURRENT_POSITIONS
            global_executor.max_drawdown_limit_usdt = Config.MAX_DRAWDOWN_LIMIT_USDT

        if global_engine and hasattr(global_engine, "yoyo"):
            if global_loop:
                if not global_engine.yoyo.worker_task or global_engine.yoyo.worker_task.done():
                    asyncio.run_coroutine_threadsafe(global_engine.yoyo.start(), global_loop)
                    exch = global_executor.exchange if (global_executor and hasattr(global_executor, 'exchange')) else None
                    if exch:
                        asyncio.run_coroutine_threadsafe(global_engine.yoyo.initialize_candles(exch), global_loop)
                        
        if global_engine:
            global_engine.symbols = Config.SYMBOLS
            global_engine.quote_denomination = Config.QUOTE_DENOMINATION
            global_engine.depth_levels = Config.LOB_DEPTH_LEVELS
            global_engine.trade_window_seconds = Config.TRADE_WINDOW_SECONDS
            global_engine.spoof_threshold_pct = Config.SPOOF_THRESHOLD_PCT
            global_engine.momentum_window_ms = Config.MOMENTUM_WINDOW_MS
            global_engine.spread_threshold_bps = Config.SPREAD_THRESHOLD_BPS
            global_engine.take_profit_bps = Config.TAKE_PROFIT_BPS
            global_engine.stop_loss_bps = Config.STOP_LOSS_BPS
            global_engine.cooldown_ms = Config.COOLDOWN_MS
            global_engine.history_file_path = Config.HISTORY_FILE_PATH
            
            if hasattr(global_engine, "yoyo") and global_engine.yoyo:
                global_engine.yoyo.symbols = Config.SYMBOLS
                global_engine.yoyo.quote_denomination = Config.QUOTE_DENOMINATION
            
            if hasattr(global_engine, "detector") and global_engine.detector:
                global_engine.detector.depth_levels = Config.LOB_DEPTH_LEVELS
                global_engine.detector.trade_window_seconds = Config.TRADE_WINDOW_SECONDS
                global_engine.detector.spoof_threshold_pct = Config.SPOOF_THRESHOLD_PCT

        log_event("⚙️ تنظیمات عمومی سیستم توسط کنترل پنل داشبورد وب با موفقیت تغییر کرد و به صورت آنی اعمال شد.")
        handler.send_json({"success": True, "message": "تنظیمات با موفقیت ذخیره و در متغیرهای محیطی ربات لود شد."})
    else:
        handler.send_json({"success": False, "message": "خطا در نوشتن تنظیمات روی فایل .env رخ داد."}, 500)
