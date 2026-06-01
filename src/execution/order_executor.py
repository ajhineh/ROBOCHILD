import os
import logging
import asyncio
from typing import Dict, Literal, Optional
import ccxt.pro as ccxt

from src.config import Config

logger = logging.getLogger("ROBORDER.OrderExecutor")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class OrderExecutor:
    """
    موتور اجرای معاملات ناهمگام فیوچرز با استفاده از CCXT Pro.
    این کلاس مسئولیت ثبت اهرم داینامیک، ارسال سفارشات مارکت، مدیریت پوزیشن‌های باز،
    و اعمال کنترل‌های ریسک پیش‌معاملاتی (PTRC) را بر عهده دارد.
    در صورت عدم ارائه کلیدهای API صرافی، سیستم به صورت هوشمند روی شبیه‌ساز (Paper Trading) اجرا می‌شود.
    """
    def __init__(
        self,
        exchange_id: str = "binance",
        live_trading: bool = False,
        max_concurrent_positions: int = 3,       # حداکثر تعداد موقعیت‌های همزمان باز
        max_drawdown_limit_usdt: float = 100.0   # سقف حد ضرر روزانه حساب
    ):
        self.exchange_id = exchange_id
        self.live_trading = live_trading
        self.max_concurrent_positions = max_concurrent_positions
        self.max_drawdown_limit_usdt = max_drawdown_limit_usdt

        # کلیدهای صرافی از متغیرهای محیطی
        self.api_key = os.getenv("EXCHANGE_API_KEY", "")
        self.secret_key = os.getenv("EXCHANGE_SECRET_KEY", "")

        self.exchange: Optional[ccxt.Exchange] = None
        self.open_positions: Dict[str, dict] = {}
        self.current_drawdown = 0.0
        self.daily_pnl = 0.0
        
        from datetime import datetime
        self.last_trade_date = datetime.now().strftime("%Y-%m-%d")

        self.setup_exchange()

    def _check_daily_reset(self) -> None:
        """بررسی تغییر روز برای بازنشانی دروداون و سود/زیان روزانه در ساعت ۲۴:۰۰"""
        from datetime import datetime
        current_date = datetime.now().strftime("%Y-%m-%d")
        if not hasattr(self, "last_trade_date") or self.last_trade_date != current_date:
            self.daily_pnl = 0.0
            self.current_drawdown = 0.0
            self.last_trade_date = current_date
            logger.info("📅 New trading day detected! Resetting daily drawdown and PnL counters to $0.00.")

    def setup_exchange(self) -> None:
        """راه‌اندازی صرافی زنده یا حالت شبیه‌ساز معاملاتی"""
        if self.live_trading and self.api_key and self.secret_key:
            try:
                exchange_class = getattr(ccxt, self.exchange_id)
                self.exchange = exchange_class({
                    'apiKey': self.api_key,
                    'secret': self.secret_key,
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'future',  # ورود به بازار USD-M Futures
                    }
                })
                logger.info(f"🟢 Connected to LIVE EXCHANGE: {self.exchange_id} (Futures USD-M mode active)")
            except Exception as e:
                logger.error(f"❌ Failed to connect to live exchange: {e}. Falling back to Simulation Mode.")
                self.live_trading = False
                self.exchange = None
        else:
            self.live_trading = False
            logger.info("🤖 Running in high-fidelity PAPER TRADING (Simulation Mode) - Zero risk to real funds")

    async def close_connections(self) -> None:
        """بستن ایمن اتصالات وب‌سوکت صرافی در زمان خروج ربات"""
        if self.exchange:
            await self.exchange.close()
            logger.info("🔌 Exchange connections closed.")

    async def execute_entry(
        self,
        symbol: str,
        side: Literal["long", "short"],
        amount_usdt: float,
        leverage: int,
        take_profit_quote: float,
        stop_loss_quote: float,
        entry_price: Optional[float] = None
    ) -> bool:
        """
        اجرای سفارش ورود مارکت به همراه تنظیم اهرم داینامیک و کنترل ریسک پیش‌معاملاتی (PTRC).
        """
        # ۱. کنترل ریسک پیش‌معاملاتی (PTRC)
        self._check_daily_reset()
        if len(self.open_positions) >= self.max_concurrent_positions:
            if "MAX_CONCURRENT_POSITIONS_CHECK" not in Config.BYPASSED_FILTERS_SET:
                logger.warning(f"⚠️ Blocked Entry for {symbol}: Maximum concurrent positions limit reached ({self.max_concurrent_positions})")
                return False

        if symbol in self.open_positions:
            logger.warning(f"⚠️ Blocked Entry: Position already open for {symbol}")
            return False

        if self.current_drawdown >= self.max_drawdown_limit_usdt:
            if "MAX_DRAWDOWN_LIMIT_CHECK" not in Config.BYPASSED_FILTERS_SET:
                logger.error(f"🚨 Blocked Entry: Daily account drawdown limit reached (${self.current_drawdown:.2f} >= ${self.max_drawdown_limit_usdt:.2f})")
                return False

        # کنترل موجودی مارجین آزاد حساب
        required_margin = (Config.TRADE_CAPITAL_PCT / 100.0) * Config.CURRENT_BALANCE
        used_margin = sum([pos.get("amount", 0.0) for pos in self.open_positions.values()])
        available_margin = Config.CURRENT_BALANCE - used_margin
        if available_margin < required_margin:
            if "AVAILABLE_MARGIN_CHECK" not in Config.BYPASSED_FILTERS_SET:
                logger.warning(f"⚠️ Blocked Entry for {symbol}: Insufficient Available Margin (${available_margin:.2f} available < required ${required_margin:.2f})")
                return False

        logger.info(f"⚡ PTRC Passed. Proceeding to execute entry order for {symbol} ({side.upper()}) | Size: ${amount_usdt} | Leverage: {leverage}x")

        # ۲. سناریو ترید زنده روی صرافی
        if self.live_trading and self.exchange:
            try:
                # الف. تنظیم اهرم معامله در صرافی
                await self.exchange.set_leverage(leverage, symbol)
                logger.info(f"✅ Set dynamic leverage to {leverage}x for {symbol} on exchange")

                # ب. تبدیل حجم معامله بر پایه USDT به مقدار توکن مورد نظر
                # برای مثال درPOP CAT ابتدا قیمت لحظه‌ای را دریافت می‌کنیم
                ticker = await self.exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                token_amount = (amount_usdt * leverage) / current_price

                # ج. ثبت سفارش مارکت صرافی
                order_side = "buy" if side == "long" else "sell"
                logger.info(f"🛒 Sending Market Order to Exchange: {order_side.upper()} {token_amount:.4f} {symbol}")
                
                order = await self.exchange.create_market_order(symbol, order_side, token_amount)
                logger.info(f"🎉 Trade Executed on Exchange. Order ID: {order['id']}")

                # ثبت موقعیت در پوزیشن‌های باز ربات
                self.open_positions[symbol] = {
                    "id": order["id"],
                    "side": side,
                    "leverage": leverage,
                    "entry_price": current_price,
                    "token_amount": token_amount,
                    "amount": amount_usdt,
                    "tp": take_profit_quote,
                    "sl": stop_loss_quote
                }
                return True

            except Exception as e:
                logger.error(f"❌ Exchange Execution Failed: {e}")
                return False

        # ۳. سناریو شبیه‌ساز (Paper Trading Mode)
        else:
            # شبیه‌سازی تاخیر صرافی (۵۰ میلی‌ثانیه)
            await asyncio.sleep(0.05)
            
            # استفاده از قیمت لحظه‌ای واقعی ثبت‌شده در معامله جهت محاسبه دقیق سود و زیان
            simulated_entry_price = entry_price if entry_price is not None else 1.0
            
            self.open_positions[symbol] = {
                "id": "mock_order_" + str(int(asyncio.get_event_loop().time() * 1000)),
                "side": side,
                "leverage": leverage,
                "entry_price": simulated_entry_price,
                "token_amount": (amount_usdt * leverage) / simulated_entry_price,
                "amount": amount_usdt,
                "tp": take_profit_quote,
                "sl": stop_loss_quote
            }
            logger.info(f"🎉 Simulated Trade Opened at ${simulated_entry_price:.6f}. Target SL: {stop_loss_quote:.6f} | Target TP: {take_profit_quote:.6f}")
            return True

    async def execute_exit(
        self,
        symbol: str,
        pnl_usdt: float,
        reason: str
    ) -> bool:
        """
        اجرای سفارش خروج مارکت برای بستن پوزیشن باز و به‌روزرسانی آمارهای روزانه ریسک.
        """
        self._check_daily_reset()
        position = self.open_positions.get(symbol)
        if not position:
            logger.warning(f"⚠️ No active open position found to exit for {symbol}")
            return False

        logger.info(f"⚡ Initiating exit order for {symbol} | Reason: {reason} | Expected PnL: ${pnl_usdt:+.4f} USDT")

        # ۱. سناریو ترید زنده روی صرافی
        if self.live_trading and self.exchange:
            try:
                # سفارش برعکس جهت ورود برای بستن پوزیشن
                exit_side = "sell" if position["side"] == "long" else "buy"
                token_amount = position["token_amount"]

                logger.info(f"🛒 Sending Exit Market Order to Exchange: {exit_side.upper()} {token_amount:.4f} {symbol}")
                order = await self.exchange.create_market_order(symbol, exit_side, token_amount)
                logger.info(f"🎉 Position Closed on Exchange. Exit Order ID: {order['id']}")

            except Exception as e:
                logger.error(f"❌ Exchange Exit Order Failed: {e}. Positions will be forcefully closed locally.")

        # ۲. اعمال تغییرات سود و ضرر در ردیاب دروداون روزانه بر اساس کل سود/زیان خالص روزانه
        self.daily_pnl += pnl_usdt
        if self.daily_pnl >= 0:
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = abs(self.daily_pnl)

        # به‌روزرسانی و ذخیره‌سازی ماندگار موجودی کل حساب (CURRENT_BALANCE)
        from src.config import Config, save_env_values
        new_balance = Config.CURRENT_BALANCE + pnl_usdt
        Config.CURRENT_BALANCE = new_balance
        save_env_values({"CURRENT_BALANCE": f"{new_balance:.4f}"})

        # پاک‌سازی موقعیت از حافظه محلی
        if symbol in self.open_positions:
            del self.open_positions[symbol]

        logger.info(f"🚪 Position Closed for {symbol}. Current Daily Drawdown: ${self.current_drawdown:.2f} | New Balance: ${new_balance:.4f}")
        return True
