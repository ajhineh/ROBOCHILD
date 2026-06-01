import asyncio
import os
import sys
import io

# پیکربندی خروجی ترمینال برای پشتیبانی از کاراکترهای فارسی در ویندوز
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


# تلاش برای وارد کردن کتابخانه ccxt
try:
    import ccxt.pro as ccxt
except ImportError:
    print("کتابخانه CCXT نصب نیست. برای نصب دستور زیر را اجرا کنید:")
    print("pip install ccxt")
    sys.exit(1)

async def main():
    """
    نمونه اولیه ربات معاملات فیوچرز بر اساس شاخص عدم تعادل دفترچه سفارشات (OBI)
    """
    # مقداردهی اولیه صرافی بایننس فیوچرز (Binance USDS-M Futures)
    # در صورت واقعی بودن، کلیدهای API را در این قسمت وارد کنید.
    exchange = ccxt.binanceusdm({
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future', # مشخص کردن نوع بازار فیوچرز
        }
    })
    
    symbol = 'BTC/USDT'
    depth_levels = 5  # تعداد سطوح عمق بازار جهت محاسبات OBI
    
    print("=" * 60)
    print(f"شروع اتصال به دفترچه سفارشات زنده صرافی بایننس برای جفت‌ارز {symbol}...")
    print(f"محاسبه بر اساس {depth_levels} سطح اول خریداران و فروشندگان انجام می‌شود.")
    print("=" * 60)
    
    try:
        while True:
            # دریافت زنده دفترچه سفارشات از طریق WebSocket (توسط CCXT Pro)
            orderbook = await exchange.watch_order_book(symbol)
            
            bids = orderbook['bids']  # سفارشات خرید [[price, volume], ...]
            asks = orderbook['asks']  # سفارشات فروش [[price, volume], ...]
            
            # بررسی اینکه آیا به اندازه کافی سفارش در دفترچه وجود دارد یا خیر
            if len(bids) < depth_levels or len(asks) < depth_levels:
                continue
                
            # ۱. محاسبه مجموع حجم خریداران و فروشندگان در سطوح قیمتی مشخص شده
            sum_bid_volume = sum([bid[1] for bid in bids[:depth_levels]])
            sum_ask_volume = sum([ask[1] for ask in asks[:depth_levels]])
            
            # ۲. محاسبه شاخص عدم تعادل دفترچه سفارشات (Order Book Imbalance - OBI)
            # این فرمول عددی بین 1- تا 1+ تولید می‌کند
            obi = (sum_bid_volume - sum_ask_volume) / (sum_bid_volume + sum_ask_volume)
            
            # ۳. قیمت لحظه‌ای (قیمت میانه یا Mid Price)
            mid_price = (bids[0][0] + asks[0][0]) / 2
            
            # ۴. تولید سیگنال آزمایشی بر اساس مقدار OBI
            # آستانه فرضی 0.3 برای گرفتن پوزیشن صعودی یا نزولی
            threshold = 0.3
            signal = "🔴 بدون موقعیت (FLAT)"
            
            if obi > threshold:
                signal = "🟢 پیشنهاد لانگ (BULLISH PRESSURE)"
            elif obi < -threshold:
                signal = "🚨 پیشنهاد شورت (BEARISH PRESSURE)"
                
            # چاپ خروجی به صورت تمیز و لحظه‌ای
            sys.stdout.write(
                f"\rقیمت: {mid_price:.2f} | OBI: {obi:+.4f} | "
                f"حجم خرید: {sum_bid_volume:.2f} | حجم فروش: {sum_ask_volume:.2f} | "
                f"وضعیت سیگنال: {signal}"
            )
            sys.stdout.flush()
            
            # جهت جلوگیری از پر شدن بیش از حد رم و پردازش خیلی سنگین در هر میلی‌ثانیه، 
            # اتصال وب‌سوکت خودکار به‌روزرسانی بعدی را دریافت می‌کند.
            
    except KeyboardInterrupt:
        print("\nبرنامه با درخواست کاربر متوقف شد.")
    except Exception as e:
        print(f"\nخطایی رخ داد: {e}")
    finally:
        await exchange.close()

if __name__ == '__main__':
    # اجرای حلقه ناهمگام پایتون
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
