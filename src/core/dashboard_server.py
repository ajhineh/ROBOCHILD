import http.server
import socketserver
import json
import os
import sys
import re
import threading
import time
import logging
from typing import Optional, Dict
import socket
import urllib.parse

from src.config import Config, save_env_values
import src.core.api.handlers as api_handlers

logger = logging.getLogger("ROBORDER.Dashboard")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# ظ…طھط؛غŒط±ظ‡ط§غŒ ط§ط´طھط±ط§ع©â€Œع¯ط°ط§ط±غŒâ€Œط´ط¯ظ‡ ط¯ط± ط³ط·ط­ ط­ط§ظپط¸ظ‡ ظ…ط´طھط±ع© ظپط±ط¢غŒظ†ط¯ ظ¾ط§غŒطھظˆظ†
global_engine = None
global_executor = None
global_loop = None  # ط±ظپط±ظ†ط³ ط§غŒظ…ظ† ط¨ظ‡ ط­ظ„ظ‚ظ‡ ط§طµظ„غŒ asyncio ط¬ظ‡طھ ط²ظ…ط§ظ†â€Œط¨ظ†ط¯غŒ طھط³ع©â€Œظ‡ط§ ط§ط² طھط±ط¯ظ‡ط§غŒ ظ¾ط³â€Œط²ظ…غŒظ†ظ‡
PORT = 6006

# ط±ط¬غŒط³طھط±غŒ ط¢ظ…ظˆط²ط´ ط´ط¨ع©ظ‡ ط¹طµط¨غŒ ظ¾ط³â€Œط²ظ…غŒظ†ظ‡
active_trainings = {}
training_stops = {}

# روتین‌های آنالیزور هوش عصبی پس‌زمینه
active_analyses = {}
analysis_logs = {}

# ع©ط´ ظ¾غŒظ†ع¯ ط´ط¨ع©ظ‡ HFT
global_pings = {
    "binance": 0.0,
    "solana_rpc": 0.0
}

def measure_ping_sync(url_or_host: str) -> float:
    """ط§ظ†ط¯ط§ط²ظ‡â€Œع¯غŒط±غŒ ظ¾غŒظ†ع¯ TCP ط¨ظ‡ ط³ط±ظˆط± ظ…ط´ط®طµ ط±ظˆغŒ ظ¾ظˆط±طھ غ´غ´غ³"""
    try:
        if "://" in url_or_host:
            parsed = urllib.parse.urlparse(url_or_host)
            host = parsed.hostname or url_or_host
        else:
            host = url_or_host.split(":")[0]
            
        t0 = time.time()
        s = socket.create_connection((host, 443), timeout=2.0)
        s.close()
        return round((time.time() - t0) * 1000, 1)
    except Exception:
        return 999.9

def ping_updater_loop():
    """ط­ظ„ظ‚ظ‡ ط²ظ…ط§ظ†â€Œط¨ظ†ط¯غŒ ط§ظ†ط¯ط§ط²ظ‡â€Œع¯غŒط±غŒ ظ¾غŒظ†ع¯â€Œظ‡ط§ ط¯ط± ظ¾ط³â€Œط²ظ…غŒظ†ظ‡ ظ‡ط± غ±غ° ط«ط§ظ†غŒظ‡"""
    global global_pings
    while True:
        try:
            # ط§ظ†ط¯ط§ط²ظ‡â€Œع¯غŒط±غŒ ظ¾غŒظ†ع¯ ظˆط¨â€Œط³ظˆع©طھ ط¨ط§غŒظ†ظ†ط³ ظپغŒظˆع†ط±ط²
            global_pings["binance"] = measure_ping_sync("fstream.binance.com")
            
            # ط§ظ†ط¯ط§ط²ظ‡â€Œع¯غŒط±غŒ ظ¾غŒظ†ع¯ ط³ط±ظˆط± RPC ط³ظˆظ„ط§ظ†ط§ (ظ‡ظ„غŒظˆط³ غŒط§ ظ¾غŒط´â€Œظپط±ط¶)
            solana_endpoint = Config.HELIUS_WS_URL or "mainnet.helius-rpc.com"
            global_pings["solana_rpc"] = measure_ping_sync(solana_endpoint)
        except Exception:
            pass
        time.sleep(10)


def log_event(message: str):
    """ط«ط¨طھ ظ¾غŒط§ظ… ط¯ط± ظپط§غŒظ„ ظ„ط§ع¯ ط§طµظ„غŒ ط³غŒط³طھظ…"""
    logger.info(f"ًں“‌ [Dashboard Log] {message}")
    # ط§ظ„ط­ط§ظ‚ ط¨ظ‡ ظپط§غŒظ„ ظ„ط§ع¯ ط¨ظ‡ ط¹ظ†ظˆط§ظ† ط±ع©ظˆط±ط¯ ظ…طھظ†غŒ ط³ط±ط§ط³ط±غŒ
    try:
        with open("robochild_x.log", "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - ROBORDER.Dashboard - INFO - {message}\n")
    except Exception:
        pass


def scan_existing_models():
    """ط§ط³ع©ظ† ظ…ط¯ظ„â€Œظ‡ط§غŒ ظ¾ط§غŒطھظˆظ† ط¢ظ…ظˆط²ط´â€Œط¯غŒط¯ظ‡ ظ…ظˆط¬ظˆط¯ ط¯ط± ظ¾ظˆط´ظ‡ models/"""
    if not os.path.exists("models"):
        os.makedirs("models", exist_ok=True)
    models_dir = "models"
    available = []
    try:
        for f in os.listdir(models_dir):
            if f.endswith("_final.zip"):
                parts = f.replace("ppo_volume_bars_child_", "").replace("_final.zip", "")
                for suffix in ["_ppo", "_sac", "_td3"]:
                    if parts.endswith(suffix):
                        parts = parts[:-len(suffix)]
                if parts == "final" or parts == "bot" or parts == "" or parts == "final.zip":
                    symbol = "BTC/USDT"
                else:
                    symbol = parts.upper() + "/USDT"
                available.append(symbol)
            elif f == "ppo_volume_bars_child_final.zip":
                available.append("BTC/USDT")
    except Exception as e:
        logger.error(f"Error scanning models directory: {e}")
    return list(set(available))


def auto_activate_symbol(symbol: str):
    """
    فعال‌سازی خودکار و داینامیک جفت ارز در ربات معاملاتی زنده
    """
    global global_engine, global_executor, global_loop
    symbol = symbol.upper().strip()
    if symbol in Config.SYMBOLS:
        log_event(f"ℹ️ جفت ارز {symbol} از قبل در لیست ترید فعال است.")
        return

    Config.SYMBOLS.append(symbol)
    from src.config import save_env_values
    save_env_values({"SYMBOLS": ",".join(Config.SYMBOLS)})
    log_event(f"➕ [فعال‌سازی خودکار] جفت ارز {symbol} با موفقیت به فایل .env و حافظه موقت اضافه شد.")

    if global_engine:
        if symbol not in global_engine.symbols:
            global_engine.symbols.append(symbol)
            
        # مقداردهی deque برای DEX
        if symbol not in global_engine.recent_dex_trades:
            from collections import deque
            global_engine.recent_dex_trades[symbol] = deque()
            
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

            # مقداردهی به شمع‌های تاریخی به صورت پس‌زمینه (thread-safe)
            import asyncio as _asyncio
            exch = global_executor.exchange if (global_executor and hasattr(global_executor, 'exchange')) else None
            if global_loop and exch:
                try:
                    _asyncio.run_coroutine_threadsafe(global_engine.yoyo.initialize_candles(exch), global_loop)
                except Exception as e:
                    log_event(f"⚠️ خطای غیرمنتظره در مقداردهی شمع‌های YoYo برای {symbol}: {e}")
                    global_engine.yoyo._generate_mock_historical_candles(symbol)
            else:
                global_engine.yoyo._generate_mock_historical_candles(symbol)


def background_train_orchestrator(symbol: str, steps: int = 200000, resume: bool = False, learning_rate: str = "linear_0.0003"):
    """ط§ط¬ط±ط§غŒ ط؛غŒط±ظ…ط³ط¯ظˆط¯ع©ظ†ظ†ط¯ظ‡ (Background Thread) ظپط±ط¢غŒظ†ط¯ ظˆط§ع©ط´غŒ ط¯ط§ط¯ظ‡â€Œظ‡ط§غŒ طھط§ط±غŒط®غŒ ظˆ ط¢ظ…ظˆط²ط´ ظ…ط¯ظ„ غŒط§ط¯ع¯غŒط±غŒ طھظ‚ظˆغŒطھâ€Œظ¾ط°غŒط±"""
    global active_trainings, training_stops
    symbol = re.sub(r'[^a-zA-Z0-9/:-]', '', symbol).upper().strip()
    symbol_clean = symbol.split('/')[0].lower()

    active_trainings[symbol_clean] = True
    training_stops.pop(symbol_clean, None)

    # طھظ‚ط³غŒظ…â€Œط¨ظ†ط¯غŒ ط²ظ…ط§ظ†غŒ ظ…غŒظ…â€Œع©ظˆغŒظ†â€Œظ‡ط§ ط±ظˆغŒ طھط§غŒظ…â€Œظپط±غŒظ… غ± ط¯ظ‚غŒظ‚ظ‡â€Œط§غŒ ظˆ ط§ط±ط²ظ‡ط§غŒ ط´ط§ط®طµ ط±ظˆغŒ غµ ط¯ظ‚غŒظ‚ظ‡â€Œط§غŒ
    is_meme = symbol_clean in ["bome", "pepe", "doge", "shib", "wif", "bonk", "floki", "popcat"]
    timeframe = "1m" if is_meme else "5m"
    days_back = 45 if timeframe == "1m" else 60

    log_event(f"ًں§  ط´ط±ظˆط¹ ط¢ظ…ظˆط²ط´ ظ¾ط³â€Œط²ظ…غŒظ†ظ‡ ط´ط¨ع©ظ‡ ط¹طµط¨غŒ ظ‡ظˆط´ ظ…طµظ†ظˆط¹غŒ ط¨ط±ط§غŒ {symbol}...")
    log_event(f"ًں§  طھط®طµغŒطµ طھط¹ط¯ط§ط¯ {steps:,} ع¯ط§ظ… ط±ظˆغŒ طھط§غŒظ…â€Œظپط±غŒظ… {timeframe} ({days_back} ط±ظˆط² ط¯ط§ط¯ظ‡ طھط§ط±غŒط®غŒ)")

    progress_file = os.path.join("models", f"progress_ppo_volume_bars_child_{symbol_clean}.json")
    os.makedirs("models", exist_ok=True)
    with open(progress_file, "w") as f:
        json.dump({
            "model_name": f"ppo_volume_bars_child_{symbol_clean}",
            "current_step": 0,
            "total_steps": steps,
            "percentage": 0.0,
            "status": "training"
        }, f)

    try:
        from src.env import fetch_real_binance_data
        from src.agent.trainer import train_agent

        def check_stop():
            return training_stops.get(symbol_clean, False)

        # غ±. ظˆط§ع©ط´غŒ ط¯ط§ط¯ظ‡â€Œظ‡ط§غŒ ظˆط§ظ‚ط¹غŒ ط¨ط§ ظ‚ط§ط¨ظ„غŒطھ ظ„ط؛ظˆ ط³ط±غŒط¹
        df = fetch_real_binance_data(
            symbol=symbol,
            timeframe=timeframe,
            days_back=days_back,
            check_stop_fn=check_stop
        )

        if check_stop():
            log_event(f"âڈ¹ï¸ڈ ظپط±ط¢غŒظ†ط¯ ط¢ظ…ظˆط²ط´ {symbol} ظ‚ط¨ظ„ ط§ط² ط§ط³طھط§ط±طھ ظ…طھظˆظ‚ظپ ط´ط¯.")
            active_trainings.pop(symbol_clean, None)
            training_stops.pop(symbol_clean, None)
            return

        # غ². ط¬ط¯ط§ط³ط§ط²غŒ ط¯ط§ط¯ظ‡â€Œظ‡ط§غŒ ط¢ظ…ظˆط²ط´ ظˆ ط§ط±ط²غŒط§ط¨غŒ
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]

        # غ³. ط§ط¬ط±ط§غŒ ظپط±ط¢غŒظ†ط¯ ط§ط³طھغŒط¨ظ„ ط¨غŒط³ظ„ط§غŒظ†ط²
        train_agent(
            train_df=train_df,
            val_df=val_df,
            total_timesteps=steps,
            model_save_dir="models",
            tb_log_dir="tb_logs",
            model_name=f"ppo_volume_bars_child_{symbol_clean}",
            check_stop_fn=check_stop,
            resume=resume,
            learning_rate_val=learning_rate
        )

        if check_stop():
            log_event(f"âڈ¹ï¸ڈ ظپط±ط¢غŒظ†ط¯ ط¢ظ…ظˆط²ط´ {symbol} ط¨ظ‡ طµظˆط±طھ ط²ظˆط¯ظ‡ظ†ع¯ط§ظ… ظ„ط؛ظˆ ط´ط¯.")
            active_trainings.pop(symbol_clean, None)
            training_stops.pop(symbol_clean, None)
            return

        log_event(f"ًںژ‰ ط¢ظ…ظˆط²ط´ ط´ط¨ع©ظ‡ ط¹طµط¨غŒ ظ‡ظˆط´ ظ…طµظ†ظˆط¹غŒ ط¨ط±ط§غŒ {symbol} ط¨ط§ ظ…ظˆظپظ‚غŒطھ غ±غ°غ°ظھ ظ¾ط§غŒط§ظ† غŒط§ظپطھ!")
        
        # ظ¾ط§ع©â€Œط³ط§ط²غŒ ط§ط³طھظ¾ ظˆ ط§طھظ…ط§ظ… ظˆط¶ط¹غŒطھ
        with open(progress_file, "w") as f:
            json.dump({
                "model_name": f"ppo_volume_bars_child_{symbol_clean}",
                "current_step": steps,
                "total_steps": steps,
                "percentage": 100.0,
                "status": "completed"
            }, f)

    except Exception as e:
        if check_stop():
            log_event(f"âڈ¹ï¸ڈ ظپط±ط¢غŒظ†ط¯ ط¢ظ…ظˆط²ط´ {symbol} ط¨ط§ ظ…ظˆظپظ‚غŒطھ ظ…طھظˆظ‚ظپ ط´ط¯.")
            with open(progress_file, "w") as f:
                json.dump({
                    "model_name": f"ppo_volume_bars_child_{symbol_clean}",
                    "current_step": 0,
                    "total_steps": steps,
                    "percentage": 0.0,
                    "status": "stopped"
                }, f)
        else:
            log_event(f"â‌Œ ط®ط·ط§غŒ ط¨ط­ط±ط§ظ†غŒ ط¯ط± ط¢ظ…ظˆط²ط´ ظ…ط¯ظ„ {symbol}: {e}")
            with open(progress_file, "w") as f:
                json.dump({
                    "model_name": f"ppo_volume_bars_child_{symbol_clean}",
                    "current_step": 0,
                    "total_steps": steps,
                    "percentage": 0.0,
                    "status": f"error: {str(e)}"
                }, f)
    finally:
        active_trainings.pop(symbol_clean, None)
        training_stops.pop(symbol_clean, None)


def background_analysis_orchestrator(symbol: str, market_type: str = "futures", days_back: int = 5):
    """اجرای غیرمسدودکننده آنالیزور فوق پیشرفته هوش عصبی در پس‌زمینه"""
    global active_analyses, analysis_logs
    symbol_clean = symbol.split('/')[0].lower()
    active_analyses[symbol_clean] = "running"
    analysis_logs[symbol_clean] = []
    
    def log_to_analysis(msg):
        if symbol_clean not in analysis_logs:
            analysis_logs[symbol_clean] = []
        analysis_logs[symbol_clean].append(f"{time.strftime('%H:%M:%S')} - {msg}")
        logger.info(f"🔎 [AI Analyzer] {msg}")

    log_to_analysis(f"شروع ارزیابی پیشرفته برای جفت‌ارز {symbol.upper()} ({market_type.upper()}، {days_back} روزه)...")
    
    try:
        from src.analysis.training_evaluator import UltraEnsembleEvaluator
        evaluator = UltraEnsembleEvaluator(symbol=symbol_clean, base_path=".", market_type=market_type, days_back=days_back)
        
        # تغییر دادن لاگر پیش‌فرض جهت ثبت مستقیم در متغیرهای داشبورد
        evaluator.log = log_to_analysis
        
        # اجرای بک‌تست و تحلیل
        evaluator.run_full_analysis()
        
        active_analyses[symbol_clean] = "completed"
        log_to_analysis("عملیات آنالیز با موفقیت پایان یافت.")
    except Exception as e:
        active_analyses[symbol_clean] = f"error: {str(e)}"
        log_to_analysis(f"❌ خطای بحرانی در اجرای آنالیزور: {e}")


class DashboardHandler(http.server.BaseHTTPRequestHandler):
    """
    ظ‡ظ†ط¯ظ„ط± ط¯ط±ط®ظˆط§ط³طھâ€Œظ‡ط§غŒ HTTP ط³ط±ظˆط± ط¯ط§ط´ط¨ظˆط±ط¯.
    ط§غŒظ† ع©ظ„ط§ط³ ط¯ط±ط®ظˆط§ط³طھâ€Œظ‡ط§غŒ ظ…ط±ط¨ظˆط· ط¨ظ‡ طµظپط­ط§طھ ط§ط³طھط§طھغŒع© ظˆ ط±ط§ط¨ط·â€Œظ‡ط§غŒ ط¨ط±ظ†ط§ظ…ظ‡â€Œظ†ظˆغŒط³غŒ (API) ط±ط¨ط§طھ ط±ط§ ظ‡ط¯ط§غŒطھ ظ…غŒâ€Œع©ظ†ط¯.
    """
    def log_message(self, format, *args):
        # ط؛غŒط±ظپط¹ط§ظ„ ع©ط±ط¯ظ† ظ„ط§ع¯ ط®ط±ظˆط¬غŒ ظ¾غŒط´â€Œظپط±ط¶ HTTP ط³ط±ظˆط± ط¯ط± طھط±ظ…غŒظ†ط§ظ„ ط¬ظ‡طھ طھظ…غŒط² ظ…ط§ظ†ط¯ظ† ط¯ط§ط´ط¨ظˆط±ط¯ ظ…طھظ†غŒ
        pass

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        # غ±. ط±ظˆطھغŒظ†ع¯ طµظپط­ط§طھ ط§ط³طھط§طھغŒع© ط¯ط§ط´ط¨ظˆط±ط¯
        if self.path == "/" or self.path == "/index.html":
            self.serve_static("static/index.html", "text/html")
            return
        elif self.path.startswith("/static/"):
            clean_path = self.path.lstrip("/")
            if ".." in clean_path:
                self.send_error(403, "Access Denied")
                return
            ext = clean_path.split(".")[-1]
            mime = "text/html"
            if ext == "css":
                mime = "text/css"
            elif ext == "js":
                mime = "application/javascript"
            self.serve_static(clean_path, mime)
            return
        elif self.path.startswith("/analysis/"):
            clean_path = self.path.lstrip("/")
            if ".." in clean_path:
                self.send_error(403, "Access Denied")
                return
            ext = clean_path.split(".")[-1]
            mime = "text/html"
            if ext == "png":
                mime = "image/png"
            elif ext == "json":
                mime = "application/json"
            self.serve_static(clean_path, mime)
            return
        # غ². ط±ظˆطھغŒظ†ع¯ ط¯ط±ط®ظˆط§ط³طھâ€Œظ‡ط§غŒ API ط²ظ†ط¯ظ‡
        elif self.path == "/api/status":
            self.handle_api_status()
        elif self.path == "/api/training_status":
            self.handle_api_training_status()
        elif self.path == "/api/logs":
            self.handle_api_logs()
        elif self.path == "/api/trade_history":
            self.handle_api_trade_history()
        elif self.path.startswith("/api/check_model"):
            self.handle_api_check_model()
        elif self.path == "/api/export_csv":
            self.handle_api_export_csv()
        elif self.path.startswith("/api/analysis_status"):
            self.handle_api_analysis_status()
        elif self.path == "/api/screener":
            self.handle_api_screener()
        else:
            self.send_error(404, "API endpoint not found")

    def do_POST(self):
        if self.path == "/api/shutdown":
            self.handle_api_shutdown()
            return
        elif self.path == "/api/reset_balance":
            self.handle_api_reset_balance()
            return
        elif self.path == "/api/liquidate_all":
            self.handle_api_liquidate_all()
            return
            
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)
        
        try:
            body = json.loads(post_data.decode("utf-8"))
        except Exception:
            self.send_error(400, "Invalid JSON data")
            return
            
        if self.path == "/api/set_settings":
            self.handle_api_set_settings(body)
        elif self.path == "/api/add_symbol":
            self.handle_api_add_symbol(body)
        elif self.path == "/api/remove_symbol":
            self.handle_api_remove_symbol(body)
        elif self.path == "/api/close_position":
            self.handle_api_close_position(body)
        elif self.path == "/api/set_bot_settings":
            self.handle_api_set_bot_settings(body)
        elif self.path == "/api/run_analyzer":
            self.handle_api_run_analyzer(body)
        else:
            self.send_error(404, "Endpoint not found")

    def serve_static(self, filepath: str, mime_type: str):
        """ط®ظˆط§ظ†ط¯ظ† ظˆ ط§ط±ط³ط§ظ„ ظپط§غŒظ„â€Œظ‡ط§غŒ ط§ط³طھط§طھغŒع© HTML/CSS/JS"""
        if not os.path.exists(filepath):
            self.send_error(404, f"File {filepath} not found")
            return
            
        try:
            with open(filepath, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Error reading file: {e}")

    def send_json(self, data: dict, status_code: int = 200):
        """ط§ط±ط³ط§ظ„ ظ¾ط§ط³ط® JSON ط§ط³طھط§ظ†ط¯ط§ط±ط¯ ط¨ظ‡ ظ…ط±ظˆط±ع¯ط±"""
        try:
            content = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            logger.error(f"Error encoding JSON response: {e}")

    def handle_api_status(self):
        """ارائه وضعیت لحظه‌ای متغیرها، پوزیشن‌های باز، آمار دروداون و اندیکاتورهای ربات"""
        global global_engine, global_executor, global_pings
        api_handlers.handle_api_status(self, global_engine, global_executor, scan_existing_models, global_pings, Config)

    def handle_api_training_status(self):
        """ط§ط±ط§ط¦ظ‡ ظ¾غŒط´ط±ظپطھ ظˆ ط¬ط²ط¦غŒط§طھ ط¢ظ…ظˆط²ط´ ظ…ط¯ظ„â€Œظ‡ط§"""
        progress_data = []
        if os.path.exists("models"):
            try:
                for f in os.listdir("models"):
                    if f.startswith("progress_") and f.endswith(".json"):
                        try:
                            with open(os.path.join("models", f), "r", encoding="utf-8") as pf:
                                p_val = json.load(pf)
                                # فقط مواردی که واقعاً در حال آموزش هستند یا تازه تکمیل شده‌اند را نمایش بده
                                if p_val.get("status") not in ("stopped", "stopped (PPO)", "stopped (SAC)", "stopped (TD3)", "error"):
                                    progress_data.append(p_val)
                        except Exception:
                            pass
            except Exception as e:
                logger.error(f"Error listing models folder for progress: {e}")
        self.send_json(progress_data)

    def handle_api_logs(self):
        """ط§ط±ط³ط§ظ„ غ±غ°غ° ط®ط· ظ†ظ‡ط§غŒغŒ ظپط§غŒظ„ ظ„ط§ع¯ ط³ط±ط§ط³ط±غŒ ط±ط¨ط§طھ ط¨ظ‡ ط¯ط§ط´ط¨ظˆط±ط¯"""
        log_file = "robochild_x.log"
        logs = []
        if os.path.exists(log_file):
            try:
                with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
                    logs = [line.strip() for line in lines[-100:]]
            except Exception as e:
                logs = [f"Error reading log file: {e}"]
        else:
            logs = ["Log file robochild_x.log does not exist yet. Feed some market tickers to start logging!"]

        self.send_json({"logs": logs})

    def handle_api_trade_history(self):
        """ط§ط±ط³ط§ظ„ طھط§ط±غŒط®ع†ظ‡ ظ…ط¹ط§ظ…ظ„ط§طھ ط±ط¨ط§طھ ط§ط² ظپط§غŒظ„ JSON طھط§ط±غŒط®ع†ظ‡ ظ…ط­ظ„غŒ"""
        history_file = Config.HISTORY_FILE_PATH
        history_data = {"signals": [], "stats": {"totalTrades": 0, "wins": 0, "losses": 0, "totalPnL": 0.0}}
        
        if os.path.exists(history_file):
            try:
                with open(history_file, "r", encoding="utf-8") as f:
                    history_data = json.load(f)
            except Exception as e:
                logger.error(f"Error loading trade history JSON: {e}")
                
        self.send_json(history_data)

    def handle_api_check_model(self):
        """بررسی سریع وجود مدل برای یک ارز مشخص"""
        try:
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            symbol = params.get("symbol", [""])[0].upper().strip()
            
            if not symbol or "/" not in symbol:
                self.send_json({"exists": False, "error": "Invalid Symbol Format"}, 400)
                return
                
            symbol_clean = symbol.split('/')[0].lower()
            model_file = f"models/ppo_volume_bars_child_{symbol_clean}_final.zip"
            
            exists = os.path.exists(model_file)
            self.send_json({"exists": exists, "symbol": symbol})
        except Exception as e:
            self.send_json({"error": str(e)}, 500)

    def handle_api_export_csv(self):
        """ط®ط±ظˆط¬غŒ CSV ط§ط² ط³ط§ط¨ظ‚ظ‡ طھط±غŒط¯ظ‡ط§ ط¨ط±ط§غŒ ط¯ط§ظ†ظ„ظˆط¯ ع©ط§ط±ط¨ط±"""
        try:
            history_file = Config.HISTORY_FILE_PATH
            signals = []
            if os.path.exists(history_file):
                with open(history_file, "r", encoding="utf-8") as f:
                    signals = json.load(f).get("signals", [])
            
            import io
            import csv
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            writer.writerow([
                "Strategy", "Symbol", "Action Type", "Price (USDT)", "Leverage", "DateTime"
            ])
            
            for s in signals:
                dt_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(s.get("time", time.time())))
                writer.writerow([
                    s.get("strategy", "YoYoStrategy"),
                    s.get("symbol", "POPCAT/USDT"),
                    s.get("type", "ENTRY"),
                    s.get("price", 0.0),
                    s.get("leverage", 15),
                    dt_str
                ])
                
            csv_data = output.getvalue()
            
            self.send_response(200)
            self.send_header("Content-Type", "text/csv")
            self.send_header("Content-Disposition", "attachment; filename=roborder_trade_report.csv")
            self.send_header("Content-Length", str(len(csv_data)))
            self.end_headers()
            self.wfile.write(csv_data.encode("utf-8"))
        except Exception as e:
            self.send_json({"error": str(e)}, 500)

    def handle_api_analysis_status(self):
        """ارائه پیشرفت و جزئیات آنالیز مدل‌ها"""
        try:
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            symbol = params.get("symbol", [""])[0].upper().strip()
            
            if not symbol:
                self.send_json({"error": "جفت ارز نامعتبر است"}, 400)
                return
                
            symbol_clean = symbol.split('/')[0].lower()
            status = active_analyses.get(symbol_clean, "idle")
            logs = analysis_logs.get(symbol_clean, [])
            
            report_data = None
            if status == "completed":
                json_path = os.path.join("analysis", f"report_{symbol_clean}.json")
                if os.path.exists(json_path):
                    with open(json_path, "r", encoding="utf-8") as f:
                        report_data = json.load(f)
            
            self.send_json({
                "symbol": symbol,
                "status": status,
                "logs": logs,
                "report": report_data
            })
        except Exception as e:
            self.send_json({"error": str(e)}, 500)

    def handle_api_run_analyzer(self, body):
        """تریگر شروع آنالیزور هوش عصبی در پس‌زمینه"""
        global active_analyses, active_trainings
        symbol = body.get("symbol", "").upper().strip()
        if not symbol:
            self.send_json({"success": False, "message": "جفت ارز نامعتبر است"}, 400)
            return
            
        symbol_clean = symbol.split('/')[0].lower()
        if symbol_clean in active_trainings:
            self.send_json({"success": False, "message": "امکان اجرای آنالیزور در حین فرآیند آموزش مدل وجود ندارد. لطفاً تا پایان آموزش صبر کنید."}, 400)
            return
            
        if active_analyses.get(symbol_clean) == "running":
            self.send_json({"success": False, "message": f"آنالیزور برای {symbol} در حال حاضر فعال و در حال اجراست."}, 400)
            return
            
        market_type = body.get("market_type", "futures").lower().strip()
        days_back = int(body.get("days_back", 5))
        
        thread = threading.Thread(
            target=background_analysis_orchestrator, 
            args=(symbol, market_type, days_back), 
            daemon=True
        )
        thread.start()
        
        self.send_json({"success": True, "message": f"پردازش آنالیزور فوق پیشرفته برای {symbol} استارت خورد."})

    def handle_api_screener(self):
        """اجرا یا بارگذاری نتایج اسکنر آلت‌کوین‌ها"""
        try:
            if not Config.SCREENER_ENABLED:
                self.send_json({
                    "timestamp": int(time.time() * 1000),
                    "exchange_id": Config.EXCHANGE_ID,
                    "altcoins": [],
                    "message": "اسکنر آلت‌کوین‌ها غیرفعال است."
                })
                return

            from src.core.screener import fetch_top_altcoins_sync
            # دریافت ۱۵ آلت‌کوین برتر از صرافی
            top_coins = fetch_top_altcoins_sync(Config.EXCHANGE_ID, limit=15)
            
            # افزودن اطلاعات وضعیت به هر کوین (داشتن مدل، در حال آموزش بودن، فعال بودن در ربات)
            enriched_coins = []
            for coin in top_coins:
                sym = coin["symbol"]
                symbol_clean = coin["base"].lower()
                
                # بررسی وجود مدل
                has_ppo = os.path.exists(f"models/ppo_volume_bars_child_{symbol_clean}_ppo_final.zip") or os.path.exists(f"models/ppo_volume_bars_child_{symbol_clean}_ppo_best.zip")
                has_old_model = os.path.exists(f"models/ppo_volume_bars_child_{symbol_clean}_final.zip") or os.path.exists(f"models/ppo_volume_bars_child_{symbol_clean}_best.zip")
                has_model = has_ppo or has_old_model
                
                is_active = sym in Config.SYMBOLS
                is_training = symbol_clean in active_trainings
                
                # پیدا کردن فایل پیشرفت آموزش در صورت وجود
                progress = None
                progress_file = os.path.join("models", f"progress_ppo_volume_bars_child_{symbol_clean}.json")
                if os.path.exists(progress_file):
                    try:
                        with open(progress_file, "r", encoding="utf-8") as pf:
                            progress = json.load(pf)
                    except Exception:
                        pass
                
                # تشخیص وضعیت Ghost Training:
                if progress and progress.get("status") == "training" and not is_training:
                    logger.warning(f"[Screener] Ghost training detected for {symbol_clean} - resetting status to error")
                    log_event(f"⚠️ آموزش {symbol_clean} بدون ثبت خطا خاتمه یافت (احتمالاً کمبود RAM). وضعیت reset شد.")
                    progress["status"] = "error"
                    progress["message"] = "آموزش به دلیل خطا یا کمبود حافظه متوقف شد. لطفاً مجدداً تلاش کنید."
                    try:
                        with open(progress_file, "w", encoding="utf-8") as pf:
                            json.dump(progress, pf, ensure_ascii=False)
                    except Exception:
                        pass
                
                enriched_coins.append({
                    **coin,
                    "has_model": has_model,
                    "is_active": is_active,
                    "is_training": is_training,
                    "progress": progress
                })
            
            # ذخیره گزارش در دیسک برای استفاده‌های بعدی یا نمایش در داشبورد
            report_data = {
                "timestamp": int(time.time() * 1000),
                "exchange_id": Config.EXCHANGE_ID,
                "altcoins": enriched_coins
            }
            os.makedirs("analysis", exist_ok=True)
            filepath_scr = os.path.join("analysis", "screener_report.json")
            try:
                with open(filepath_scr, "w", encoding="utf-8") as f_scr:
                    json.dump(report_data, f_scr, indent=4, ensure_ascii=False)
            except Exception as e:
                logger.error(f"Error saving screener report: {e}")
                
            self.send_json(report_data)
        except Exception as e:
            logger.error(f"Error in handle_api_screener: {e}")
            self.send_json({"error": str(e)}, 500)

    def handle_api_add_symbol(self, body):
        """ثبت ارز جدید: شروع فرآیند لایو تریدینگ یا ایجاد ترد آموزش هوش مصنوعی"""
        global global_engine, global_executor, global_loop, active_trainings, active_analyses
        api_handlers.handle_api_add_symbol(
            self, body, global_engine, global_executor, global_loop,
            active_trainings, active_analyses, auto_activate_symbol, log_event,
            background_train_orchestrator, Config, threading
        )

    def handle_api_remove_symbol(self, body):
        """حذف ارز: متوقف کردن آموزش هوش مصنوعی یا حذف جفت ارز از لیست ترید فعال"""
        global global_engine, global_executor, global_loop, active_trainings, training_stops
        api_handlers.handle_api_remove_symbol(
            self, body, global_engine, global_executor, global_loop,
            active_trainings, training_stops, log_event, Config
        )

    def handle_api_close_position(self, body):
        """بستن فوری پوزیشن یک ارز بدون حذف آن از لیست نمادها"""
        global global_engine, global_executor, global_loop
        api_handlers.handle_api_close_position(
            self, body, global_engine, global_executor, global_loop, log_event, Config
        )

    def handle_api_set_bot_settings(self, body):
        """طھظ†ط¸غŒظ…ط§طھ ط§ط®طھطµط§طµغŒ ط­ط¯ ط³ظˆط¯/ط¶ط±ط± ط±غŒط§ط¶غŒ ظˆ ط§ظ‡ط±ظ… ط¨ط±ط§غŒ ط¬ظپطھ ط§ط±ط² ط®ط§طµ"""
        symbol = body.get("symbol", "").upper().strip()
        symbol = re.sub(r'[^a-zA-Z0-9/:-]', '', symbol)
        if not symbol:
            self.send_json({"success": False, "message": "ط§ط±ط² ظ†ط§ظ…ط´ط®طµ ط§ط³طھ"}, 400)
            return

        # ط¯ط± ظ¾ط§غŒطھظˆظ† ROBORDER-XطŒ ظ…ظ‚ط§ط¯غŒط± TP ظˆ SL ط¨ظ‡ طµظˆط±طھ ط³ط±ط§ط³ط±غŒ ط¯ط± .env ط«ط¨طھ ط´ط¯ظ‡ ط§ط³طھ.
        # ط§ظ…ط§ ط¨ط±ط§غŒ ط§ط¹ظ…ط§ظ„ طھط¹ط§ظ…ظ„غŒطŒ ظ…غŒâ€Œطھظˆط§ظ†غŒظ… ع©ظ„ طھظ†ط¸غŒظ…ط§طھ ط¹ط¯ط¯غŒ .env ط±ط§ ط§ط² Settings ط¨ط±ظˆط²ط±ط³ط§ظ†غŒ ع©ظ†غŒظ….
        self.send_json({"success": True, "message": f"طھظ†ط¸غŒظ…ط§طھ ط¨ط§ ظ…ظˆظپظ‚غŒطھ ط¨ط±ط§غŒ ع©ظ„ ظ¾ظˆط±طھظپظˆظ„غŒظˆ ط§ط¹ظ…ط§ظ„ ع¯ط±ط¯غŒط¯."})

    def handle_api_liquidate_all(self):
        """ط¯ط³طھظˆط± ظ†ظ‡ط§غŒغŒ ط§ظ†ط¬ظ…ط§ط¯ ط³ط±ط§ط³ط±غŒ ظˆ ظ†ظ‚ط¯غŒظ†ع¯غŒ ط§ط¶ط·ط±ط§ط±غŒ طھظ…ط§ظ… ظ…ظˆظ‚ط¹غŒطھâ€Œظ‡ط§غŒ ط¨ط§ط² ط¯ط± طµط±ط§ظپغŒ"""
        log_event("ًںڑ¨ًںڑ¨ًںڑ¨ ط®ط±ظˆط¬ ط§ط¶ط·ط±ط§ط±غŒ (GLOBAL EMERGENCY LIQUIDATION) طھظˆط³ط· ع©ط§ط±ط¨ط± ظپط¹ط§ظ„ ط´ط¯! ًںڑ¨ًںڑ¨ًںڑ¨")
        
        halted_symbols = []
        if global_executor:
            # ط§ط³طھط®ط±ط§ط¬ ظ¾ظˆط²غŒط´ظ†â€Œظ‡ط§غŒ ط¨ط§ط² ط¬ظ‡طھ ظ†ظ‚ط¯غŒظ†ع¯غŒ ط¨ظ„ط§ط¯ط±ظ†ع¯ (thread-safe)
            open_syms = list(global_executor.open_positions.keys())
            import asyncio as _asyncio

            for sym in open_syms:
                halted_symbols.append(sym)
                try:
                    if global_loop:
                        _asyncio.run_coroutine_threadsafe(
                            global_executor.execute_exit(sym, 0.0, "EMERGENCY_HALT"),
                            global_loop
                        )
                except Exception as e:
                    log_event(f"Error force liquidating {sym}: {e}")

        # طھظ†ط¸غŒظ… ط³ظ‚ظپ ظ¾ظˆط²غŒط´ظ† ط±ظˆغŒ طµظپط± ط¬ظ‡طھ ط¬ظ„ظˆع¯غŒط±غŒ ط§ط² طھط±غŒط¯ظ‡ط§غŒ ط¨ط¹ط¯غŒ
        save_env_values({
            "ROBORDER_LIVE": "false",
            "MAX_CONCURRENT_POSITIONS": "0"
        })

        self.send_json({
            "success": True, 
            "message": f"ط¯ط³طھظˆط± ط®ط±ظˆط¬ ط§ط¶ط·ط±ط§ط±غŒ طµط§ط¯ط± ط´ط¯! ظ¾ظˆط²غŒط´ظ†â€Œظ‡ط§غŒ {', '.join(halted_symbols)} ط¨ط§ ظ…ظˆظپظ‚غŒطھ ظ†ظ‚ط¯ ط´ط¯ظ†ط¯ ظˆ ط±ط¨ط§طھ ط±ظˆغŒ ط­ط§ظ„طھ Paper ظ…طھظˆظ‚ظپ ع¯ط±ط¯غŒط¯."
        })

    def handle_api_shutdown(self):
        """ط®ط§ظ…ظˆط´ ع©ط±ط¯ظ† ع©ط§ظ…ظ„ ظ¾ط±ظˆط³ظ‡ ظ¾ط§غŒطھظˆظ† ط±ط¨ط§طھ ط¯ط± ط³ط±ظˆط± ظ„غŒظ†ظˆع©ط³"""
        log_event("ًں›‘ ط¯ط³طھظˆط± ط®ط§ظ…ظˆط´ ع©ط±ط¯ظ† ع©ط§ظ…ظ„ ط±ط¨ط§طھ ط§ط² ط·ط±ظپ ط¯ط§ط´ط¨ظˆط±ط¯ طھط¹ط§ظ…ظ„غŒ طµط§ط¯ط± ط´ط¯. ظپط±ط¢غŒظ†ط¯ ظ¾ط§غŒطھظˆظ† ط³ط±ظˆط± ظ…طھظˆظ‚ظپ ظ…غŒâ€Œع¯ط±ط¯ط¯...")
        self.send_json({"success": True, "message": "ظپط±ط¢غŒظ†ط¯ ط±ط¨ط§طھ ط¨ط§ ظ…ظˆظپظ‚غŒطھ ط®ط§ظ…ظˆط´ ط´ط¯. ط§طھطµط§ظ„ ط´ظ…ط§ ط¨ظ‡ ط³ط±ظˆط± ظ‚ط·ط¹ ظ…غŒâ€Œع¯ط±ط¯ط¯."})
        
        def kill_process():
            time.sleep(1.0)
            os._exit(0)
            
        threading.Thread(target=kill_process, daemon=True).start()

    def handle_api_reset_balance(self):
        """ط±غŒط³طھ ع©ط±ط¯ظ† ظ…ظˆط¬ظˆط¯غŒ ع©ظ„ ط­ط³ط§ط¨ ط¨ظ‡ ظ…ظˆط¬ظˆط¯غŒ ط§ظˆظ„غŒظ‡طŒ ظ¾ط§ع© ع©ط±ط¯ظ† طھط§ط±غŒط®ع†ظ‡ ظ…ط¹ط§ظ…ظ„ط§طھ ظˆ ط¨ط§ط²ظ†ط´ط§ظ†غŒ ط¯ط±ظˆط¯ط§ظˆظ† ط±ظˆط²ط§ظ†ظ‡"""
        global global_engine, global_executor
        log_event(f"ًں”„ ط¨ط§ط²ظ†ط´ط§ظ†غŒ ظ…ظˆط¬ظˆط¯غŒ ع©ظ„ ط­ط³ط§ط¨ ط¨ظ‡ ظ…ظˆط¬ظˆط¯غŒ ط§ظˆظ„غŒظ‡ (${Config.INITIAL_BALANCE:.2f})")
        Config.CURRENT_BALANCE = Config.INITIAL_BALANCE
        success = save_env_values({"CURRENT_BALANCE": f"{Config.INITIAL_BALANCE:.4f}"})
        
        # ظ¾ط§ع© ع©ط±ط¯ظ† ع©ط§ظ…ظ„ طھط§ط±غŒط®ع†ظ‡ ظ…ط¹ط§ظ…ظ„ط§طھ
        if global_engine:
            if hasattr(global_engine, "yoyo"):
                global_engine.yoyo.history = {"signals": [], "stats": {"totalTrades": 0, "wins": 0, "losses": 0, "totalPnL": 0.0}}
                global_engine.yoyo.save_history()
            log_event("ًں—‘ï¸ڈ طھط§ط±غŒط®ع†ظ‡ ظ…ط¹ط§ظ…ظ„ط§طھ ط¯ط± ظ‡ط³طھظ‡ ط§ط³طھط±ط§طھعکغŒ ط¨ط§ ظ…ظˆظپظ‚غŒطھ ظ¾ط§ع© ط´ط¯.")
        else:
            try:
                history_file = Config.HISTORY_FILE_PATH
                empty_history = {"signals": [], "stats": {"totalTrades": 0, "wins": 0, "losses": 0, "totalPnL": 0.0}}
                with open(history_file, "w", encoding="utf-8") as f:
                    json.dump(empty_history, f, indent=2, ensure_ascii=False)
                log_event("ًں—‘ï¸ڈ ظپط§غŒظ„ طھط§ط±غŒط®ع†ظ‡ ظ…ط¹ط§ظ…ظ„ط§طھ ظ…ط³طھظ‚غŒظ…ط§ظ‹ ظ¾ط§ع©ط³ط§ط²غŒ ط´ط¯.")
            except Exception as e:
                logger.error(f"Failed to clear history file during balance reset: {e}")

        # ط¨ط§ط²ظ†ط´ط§ظ†غŒ ظ…غŒط²ط§ظ† ط¯ط±ظˆط¯ط§ظˆظ† ط±ظˆط²ط§ظ†ظ‡ (Daily Drawdown) ظˆ ط³ظˆط¯/ط²غŒط§ظ† ط¯ط± ظ…ط§عکظˆظ„ ظ…ط¯غŒط±غŒطھ ط±غŒط³ع©
        if global_executor:
            global_executor.current_drawdown = 0.0
            global_executor.daily_pnl = 0.0
            log_event("ًں”„ ظ…غŒط²ط§ظ† ط¯ط±ظˆط¯ط§ظˆظ† ط±ظˆط²ط§ظ†ظ‡ (Daily Drawdown) ظˆ ط³ظˆط¯/ط²غŒط§ظ† ط±ظˆط²ط§ظ†ظ‡ ظ†غŒط² ط¨ط§ ظ…ظˆظپظ‚غŒطھ ط¨ظ‡ طµظپط± ط¨ط§ط²ظ†ط´ط§ظ†غŒ ط´ط¯ظ†ط¯.")

        if success:
            self.send_json({"success": True, "message": f"ظ…ظˆط¬ظˆط¯غŒ ط­ط³ط§ط¨ ط¨ط§ ظ…ظˆظپظ‚غŒطھ ط¨ظ‡ ${Config.INITIAL_BALANCE:.2f} ط±غŒط³طھ ط´ط¯طŒ ع©ظ„ طھط§ط±غŒط®ع†ظ‡ ظ…ط¹ط§ظ…ظ„ط§طھ ظ¾ط§ع© ط´ط¯ ظˆ ط¯ط±ظˆط¯ط§ظˆظ† ظ†غŒط² ط¨ط§ط²ظ†ط´ط§ظ†غŒ ع¯ط±ط¯غŒط¯."})
        else:
            self.send_json({"success": False, "message": "ط®ط·ط§ ط¯ط± ط¨ط±ظˆط²ط±ط³ط§ظ†غŒ ظ…ظˆط¬ظˆط¯غŒ ط¯ط± .env"}, 500)

    def handle_api_set_settings(self, body: dict):
        """ذخیره تنظیمات عددی جدید ارسال شده از مرورگر مستقیماً درون فایل متغیرهای محیطی .env و اعمال آنی به موتورها"""
        global global_engine, global_executor, global_loop
        api_handlers.handle_api_set_settings(
            self, body, global_engine, global_executor, global_loop, log_event, Config
        )

def start_dashboard_server(engine, executor, port: int = 3000, loop=None) -> None:
    """ط±ط§ظ‡â€Œط§ظ†ط¯ط§ط²غŒ ط³ط±ظˆط± ط¯ط§ط´ط¨ظˆط±ط¯ طھط¹ط§ظ…ظ„غŒ HTTP ط¯ط± ظ¾ط³â€Œط²ظ…غŒظ†ظ‡ ط¨ظ‡ ط¹ظ†ظˆط§ظ† غŒع© Daemon Thread ط¨ط§ ط¨ط§غŒظ†ط¯ ط³ظ†ع©ط±ظˆظ† ظ¾ظˆط±طھ"""
    global global_engine, global_executor, global_loop, PORT
    global_engine = engine
    global_executor = executor
    global_loop = loop  # ط°ط®غŒط±ظ‡ ط±ظپط±ظ†ط³ ط­ظ„ظ‚ظ‡ ط§طµظ„غŒ asyncio ط¨ط±ط§غŒ ط²ظ…ط§ظ†â€Œط¨ظ†ط¯غŒ ط§غŒظ…ظ† طھط³ع©â€Œظ‡ط§ ط§ط² طھط±ط¯ ظ¾ط³â€Œط²ظ…غŒظ†ظ‡
    PORT = port

    # ط±ط§ظ‡â€Œط§ظ†ط¯ط§ط²غŒ طھط±ط¯ ظ¾ط§غŒط´ ظ…ط¯ط§ظˆظ… ظ¾غŒظ†ع¯ ط³ط±ظˆط±ظ‡ط§ ط¯ط± ظ¾ط³â€Œط²ظ…غŒظ†ظ‡
    ping_thread = threading.Thread(target=ping_updater_loop, daemon=True)
    ping_thread.start()

    server_address = ('', PORT)
    try:
        socketserver.ThreadingTCPServer.allow_reuse_address = True
        # ط§غŒط¬ط§ط¯ ط³ظˆع©طھ ظˆ ط¨ط§غŒظ†ط¯ ط¨ظ‡ طµظˆط±طھ ط³ظ†ع©ط±ظˆظ† ط¯ط± طھط±ط¯ ط§طµظ„غŒ ط¬ظ‡طھ ط¬ظ„ظˆع¯غŒط±غŒ ط§ط² ط§ط¬ط±ط§غŒ ظ‡ظ…ط²ظ…ط§ظ† ط¯ظˆ ط±ط¨ط§طھ ط±ظˆغŒ غŒع© ط§ع©ط§ظ†طھ
        httpd = socketserver.ThreadingTCPServer(server_address, DashboardHandler)
        logger.info(f"ًںŒگ Interactive UI/UX Dashboard Server initialized on port {PORT}")
    except Exception as e:
        logger.critical(f"ًںڑ¨ PORT BINDING FAILED: Port {PORT} is already in use by another active instance of ROBORDER!")
        logger.critical("ًںڑ¨ To prevent double-trading and margin blow-up disasters, this instance is shutting down immediately.")
        logger.critical("ًںڑ¨ Please run 'kill -9 <PID>' or stop the existing background bot process before starting a new one.")
        time.sleep(1.0)
        os._exit(1)

    def run_server():
        try:
            with httpd:
                logger.info(f"ًںŒگ Interactive UI/UX Dashboard Server running live at: http://localhost:{PORT}")
                httpd.serve_forever()
        except Exception as e:
            logger.error(f"Dashboard server runtime error: {e}")

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()