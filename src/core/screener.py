import logging
import os
import json
import time
import ccxt
from typing import List, Dict

logger = logging.getLogger("ROBORDER.Screener")

def fetch_top_altcoins_sync(exchange_id: str, limit: int = 15) -> List[Dict]:
    """
    استعلام و فیلتر کردن آلت‌کوین‌های باکیفیت و با نقدینگی بالا مستقیماً از صرافی
    """
    try:
        # ساخت نمونه همگام از صرافی برای دریافت دیتای تیکرها
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future'
            }
        })
        
        logger.info(f"🔍 Fetching tickers from {exchange_id}...")
        tickers = exchange.fetch_tickers()
        
        # فیلتر کردن استیبل‌کوین‌ها و ارزهای فرعی
        stablecoins = {'usdt', 'usdc', 'busd', 'tusd', 'dai', 'eur', 'gbp', 'aud', 'fdusd', 'try', 'rub'}
        
        filtered_tickers = []
        for symbol, ticker in tickers.items():
            if not ticker or ticker.get('quoteVolume') is None:
                continue
                
            # استخراج نام دارایی پایه و دارایی کوت
            if ':' in symbol:
                # قالب فیوچرز: BASE/QUOTE:QUOTE
                parts = symbol.split(':')
                base_quote = parts[0].split('/')
                base = base_quote[0].lower()
                quote = base_quote[1].lower()
            elif '/' in symbol:
                parts = symbol.split('/')
                base = parts[0].lower()
                quote = parts[1].lower()
            else:
                continue
                
            # ما فقط جفت‌ارزهایی که ارز کوت آن‌ها تتر (USDT) است را ترید می‌کنیم
            if quote != 'usdt':
                continue
                
            # فیلتر کردن استیبل‌کوین‌ها
            if base in stablecoins:
                continue
                
            # فیلتر کردن قراردادهای زمان‌دار (فقط دائمی / swap یا perpetual)
            info = ticker.get('info', {})
            if isinstance(info, dict):
                contract_type = info.get('contractType', '').lower()
                if contract_type and 'perpetual' not in contract_type:
                    continue
            
            # ثبت در لیست فیلتر شده‌ها
            filtered_tickers.append({
                "symbol": symbol,
                "base": base.upper(),
                "quoteVolume": ticker['quoteVolume'],
                "percentage": ticker.get('percentage', 0.0),
                "close": ticker.get('close', 0.0)
            })
            
        # مرتب‌سازی بر اساس حجم ۲۴ ساعته (نزولی)
        filtered_tickers.sort(key=lambda x: x['quoteVolume'], reverse=True)
        
        # انتخاب دارایی‌های برتر
        top_altcoins = filtered_tickers[:limit]
        logger.info(f"✅ Successfully screened top {len(top_altcoins)} high-liquidity altcoins.")
        return top_altcoins
        
    except Exception as e:
        logger.error(f"❌ Error screening altcoins: {e}")
        return []

def generate_screener_report(exchange_id: str, limit: int = 15) -> str:
    """
    اجرای اسکنر و ذخیره‌سازی خروجی به صورت گزارش JSON
    """
    top_coins = fetch_top_altcoins_sync(exchange_id, limit)
    report_data = {
        "timestamp": int(time.time() * 1000),
        "exchange_id": exchange_id,
        "altcoins": top_coins
    }
    
    os.makedirs("analysis", exist_ok=True)
    filepath = os.path.join("analysis", "screener_report.json")
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=4, ensure_ascii=False)
        logger.info(f"💾 Saved screener report to {filepath}")
    except Exception as e:
        logger.error(f"❌ Error saving screener report file: {e}")
        
    return filepath
