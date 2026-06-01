import asyncio
import sys
import io
from collections import deque

# پیکربندی خروجی ترمینال برای پشتیبانی از کاراکترهای فارسی در ویندوز
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


# تلاش برای وارد کردن کتابخانه ccxt.pro
try:
    import ccxt.pro as ccxt
except ImportError:
    print("کتابخانه CCXT Pro نصب نیست. جهت نصب:")
    print("pip install ccxt")
    sys.exit(1)

# ساختار ذخیره معاملات اخیر برای محاسبه جریان سفارشات فعال (Order Flow)
# ذخیره حجم معاملات خرید و فروش مارکت در یک پنجره زمانی مشخص
trade_window_seconds = 10
recent_trades = deque()

async def track_trades(exchange, symbol):
    """
    وظیفه (Task) اول: دریافت بلادرنگ معاملات نهایی شده (Market Trades)
    این جریان صد درصد واقعی است و قابل لغو یا جعل (Spoofing) نیست.
    """
    global recent_trades
    print(f"[WebSocket] اتصال به جریان معاملات (Trades Stream) برای {symbol} برقرار شد.")
    
    try:
        while True:
            trades = await exchange.watch_trades(symbol)
            for trade in trades:
                # هر معامله شامل: زمان، قیمت، حجم و جهت (buy یعنی خریدار مارکت / sell یعنی فروشنده مارکت)
                trade_info = {
                    'timestamp': trade['timestamp'],
                    'price': trade['price'],
                    'amount': trade['amount'],
                    'side': trade['side']  # 'buy' یا 'sell'
                }
                recent_trades.append(trade_info)
                
            # پاک‌سازی معاملات قدیمی‌تر از پنجره زمانی مورد نظر
            now_ms = exchange.milliseconds()
            while recent_trades and (now_ms - recent_trades[0]['timestamp'] > trade_window_seconds * 1000):
                recent_trades.popleft()
                
    except Exception as e:
        print(f"\n[خطا در جریان معاملات]: {e}")

async def analyze_order_book_and_detect_spoofing(exchange, symbol):
    """
    وظیفه (Task) دوم: دریافت بلادرنگ دفترچه سفارشات و تطبیق آن با جریان معاملات
    جهت تفکیک دیوارهای خرید/فروش واقعی از دیوارهای کاذب (Spoofing)
    """
    global recent_trades
    print(f"[WebSocket] اتصال به دفترچه سفارشات (Order Book) برای {symbol} برقرار شد.")
    
    depth_levels = 5
    
    try:
        # متغیرهایی برای ذخیره حجم‌های دفترچه در گام قبلی جهت محاسبه نرخ لغو/جذب
        prev_bid_vol = None
        prev_ask_vol = None
        
        while True:
            orderbook = await exchange.watch_order_book(symbol)
            bids = orderbook['bids']
            asks = orderbook['asks']
            
            if len(bids) < depth_levels or len(asks) < depth_levels:
                continue
                
            # محاسبه حجم در سطوح بالای دفترچه سفارشات (Passive Liquidity)
            current_bid_vol = sum([bid[1] for bid in bids[:depth_levels]])
            current_ask_vol = sum([ask[1] for ask in asks[:depth_levels]])
            mid_price = (bids[0][0] + asks[0][0]) / 2
            
            # محاسبه کل معاملات خرید و فروش مارکت انجام شده در ۱۰ ثانیه اخیر (Active Executed Volume)
            market_buy_vol = sum([t['amount'] for t in recent_trades if t['side'] == 'buy'])
            market_sell_vol = sum([t['amount'] for t in recent_trades if t['side'] == 'sell'])
            
            # محاسبه شاخص عدم تعادل دفترچه سفارشات (خام)
            raw_obi = (current_bid_vol - current_ask_vol) / (current_bid_vol + current_ask_vol)
            
            # تطبیق حجم سفارشات با جریان معاملات نهایی شده (سیستم هوشمند مقابله با Spoofing)
            # اگر حجم زیادی از دفترچه کم شود بدون اینکه معامله مارکتی معادل آن ثبت شده باشد -> نشان‌دهنده لغو سفارش (Spoofing) است.
            spoof_detected = False
            spoof_info = "پایدار (No Spoofing)"
            
            if prev_bid_vol is not None and prev_ask_vol is not None:
                delta_bid_vol = prev_bid_vol - current_bid_vol
                delta_ask_vol = prev_ask_vol - current_ask_vol
                
                # اگر حجم بید کاهش یافته ولی معامله فروش سنگینی رخ نداده باشد:
                if delta_bid_vol > 0 and (delta_bid_vol - market_sell_vol) > (prev_bid_vol * 0.15):
                    # بیش از ۱۵ درصد حجم بید بدون معامله معادل لغو شده است
                    spoof_detected = True
                    spoof_info = "⚠️ لغو سفارش خرید سنگین (Buy Spoofing Detected)"
                
                # اگر حجم اسک کاهش یافته ولی معامله خرید سنگینی رخ نداده باشد:
                elif delta_ask_vol > 0 and (delta_ask_vol - market_buy_vol) > (prev_ask_vol * 0.15):
                    # بیش از ۱۵ درصد حجم اسک بدون معامله معادل لغو شده است
                    spoof_detected = True
                    spoof_info = "⚠️ لغو سفارش فروش سنگین (Sell Spoofing Detected)"

            # ذخیره مقادیر برای فریم بعدی
            prev_bid_vol = current_bid_vol
            prev_ask_vol = current_ask_vol
            
            # تصمیم‌گیری برای تولید سیگنال نهایی با ترکیب OBI عددی و تاییدیه جریان معاملات
            # فیلتر سیگنال: اگر OBI مثبت است ولی بید اسپوفینگ داریم، پوزیشن لانگ فیلتر می‌شود.
            final_signal = "🔴 بدون موقعیت (FLAT)"
            
            if raw_obi > 0.3:
                if "Buy Spoofing" in spoof_info:
                    final_signal = "🔒 فیلتر شده (تلاش برای فریب خریدار - سیگنال لانگ رد شد)"
                else:
                    # تایید لانگ به شرط وجود خریدار مارکت واقعی در ۱۰ ثانیه اخیر
                    if market_buy_vol > market_sell_vol:
                        final_signal = "🟢 خرید تایید شده (CONFIRMED LONG - Genuine Buying)"
                    else:
                        final_signal = "⏳ خنثی (عدم تایید خریدار مارکت)"
            
            elif raw_obi < -0.3:
                if "Sell Spoofing" in spoof_info:
                    final_signal = "🔒 فیلتر شده (تلاش برای فریب فروشنده - سیگنال شورت رد شد)"
                else:
                    # تایید شورت به شرط وجود فروشنده مارکت واقعی در ۱۰ ثانیه اخیر
                    if market_sell_vol > market_buy_vol:
                        final_signal = "🚨 فروش تایید شده (CONFIRMED SHORT - Genuine Selling)"
                    else:
                        final_signal = "⏳ خنثی (عدم تایید فروشنده مارکت)"

            # نمایش اطلاعات جامع خروجی
            sys.stdout.write(
                f"\rقیمت: {mid_price:.4f} | OBI خام: {raw_obi:+.2f} | "
                f"معاملات خرید ۱۰ث: {market_buy_vol:.1f} | معاملات فروش ۱۰ث: {market_sell_vol:.1f} | "
                f"وضعیت فریب: {spoof_info:<35} | سیگنال: {final_signal:<40}"
            )
            sys.stdout.flush()
            
            # تعلیق بسیار کوتاه جهت همگام‌سازی
            await asyncio.sleep(0.01)
            
    except Exception as e:
        print(f"\n[خطا در تحلیل دفترچه]: {e}")

async def main():
    # راه‌اندازی صرافی بایننس فیوچرز
    exchange = ccxt.binanceusdm({
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future',
        }
    })
    
    symbol = 'POPCAT/USDT'  # جفت‌ارز درخواستی شما به عنوان مثال
    
    # اجرای موازی جریان معاملات و تحلیل دفترچه سفارشات با ساختار Async
    try:
        await asyncio.gather(
            track_trades(exchange, symbol),
            analyze_order_book_and_detect_spoofing(exchange, symbol)
        )
    except KeyboardInterrupt:
        print("\nتوقف ربات توسط کاربر.")
    finally:
        await exchange.close()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
