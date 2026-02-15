#!/usr/bin/env python3
"""
AMA Pro Scanner - Manual Symbol List Edition
For when your IP is temporarily banned from Binance
"""

import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime
from typing import List, Optional

# Popular USDT futures pairs - update this list as needed
POPULAR_SYMBOLS = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT',
    'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT', 'MATIC/USDT', 'DOT/USDT',
    'LTC/USDT', 'LINK/USDT', 'UNI/USDT', 'ATOM/USDT', 'XLM/USDT',
    'NEAR/USDT', 'ALGO/USDT', 'BCH/USDT', 'FIL/USDT', 'APT/USDT',
    'ARB/USDT', 'OP/USDT', 'AAVE/USDT', 'GRT/USDT', 'SUI/USDT',
    'SEI/USDT', 'TIA/USDT', 'INJ/USDT', 'RUNE/USDT', 'FTM/USDT',
    'SAND/USDT', 'MANA/USDT', 'AXS/USDT', 'GALA/USDT', 'ICP/USDT',
    'ETC/USDT', 'CRV/USDT', 'LDO/USDT', 'MKR/USDT', 'SNX/USDT',
    'COMP/USDT', 'YFI/USDT', 'SUSHI/USDT', '1INCH/USDT', 'ENJ/USDT',
    'CHZ/USDT', 'ZIL/USDT', 'BAT/USDT', 'ZRX/USDT', 'OMG/USDT',
    'ROSE/USDT', 'KAVA/USDT', 'CELO/USDT', 'ONE/USDT', 'ZEC/USDT',
    'DASH/USDT', 'XMR/USDT', 'EOS/USDT', 'THETA/USDT', 'VET/USDT',
    'TRX/USDT', 'FET/USDT', 'AGIX/USDT', 'OCEAN/USDT', 'RNDR/USDT',
    'HBAR/USDT', 'QNT/USDT', 'FLOW/USDT', 'EGLD/USDT', 'XTZ/USDT',
    'MINA/USDT', 'APE/USDT', 'IMX/USDT', 'BLUR/USDT', 'PEPE/USDT',
    'WLD/USDT', 'STX/USDT', 'KAS/USDT', 'ORDI/USDT', 'JTO/USDT',
    'PYTH/USDT', 'BONK/USDT', 'FLOKI/USDT', 'SHIB/USDT', 'WIF/USDT',
    'PENDLE/USDT', 'JUP/USDT', 'DYM/USDT', 'STRK/USDT', 'ONDO/USDT',
    'MEME/USDT', 'BOME/USDT', 'ENA/USDT', 'W/USDT', 'ETHFI/USDT',
    'REZ/USDT', 'SAGA/USDT', 'TAO/USDT', 'OMNI/USDT', 'NOT/USDT'
]

class AMAScanner:
    def __init__(self):
        # Initialize Binance with rate limiting
        self.exchange = ccxt.binance({
            'enableRateLimit': True,  # CCXT handles rate limiting
            'options': {
                'defaultType': 'future',  # Use futures
            }
        })
        
    def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 300) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data"""
        try:
            # CCXT automatically handles rate limiting
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            # Silently skip errors for individual symbols
            return None
    
    def calculate_ama_signals(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[dict]:
        """Calculate AMA crossover signals"""
        if df is None or len(df) < 50:
            return []
        
        signals = []
        
        # Calculate EMAs for AMA Pro logic
        df['ema8'] = df['close'].ewm(span=8, adjust=False).mean()
        df['ema14'] = df['close'].ewm(span=14, adjust=False).mean()
        df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema55'] = df['close'].ewm(span=55, adjust=False).mean()
        
        # Calculate RSI for filtering
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Check for crossovers in last few candles
        for i in range(-5, 0):
            try:
                # Fast MA crossover slow MA
                fast_ma_current = df['ema14'].iloc[i]
                fast_ma_prev = df['ema14'].iloc[i-1]
                slow_ma_current = df['ema55'].iloc[i]
                slow_ma_prev = df['ema55'].iloc[i-1]
                
                rsi = df['rsi'].iloc[i]
                
                # Long signal: fast crosses above slow
                if fast_ma_prev <= slow_ma_prev and fast_ma_current > slow_ma_current:
                    angle = self.calculate_angle(df, i)
                    if abs(angle) > 5:  # Filter weak signals
                        signals.append({
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'type': 'LONG',
                            'angle': round(angle, 2),
                            'price': round(df['close'].iloc[i], 6),
                            'rsi': round(rsi, 1),
                            'time': df.index[i].strftime('%Y-%m-%d %H:%M')
                        })
                
                # Short signal: fast crosses below slow
                elif fast_ma_prev >= slow_ma_prev and fast_ma_current < slow_ma_current:
                    angle = self.calculate_angle(df, i)
                    if abs(angle) > 5:  # Filter weak signals
                        signals.append({
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'type': 'SHORT',
                            'angle': round(angle, 2),
                            'price': round(df['close'].iloc[i], 6),
                            'rsi': round(rsi, 1),
                            'time': df.index[i].strftime('%Y-%m-%d %H:%M')
                        })
                        
            except:
                continue
        
        return signals
    
    def calculate_angle(self, df: pd.DataFrame, idx: int) -> float:
        """Calculate crossover angle"""
        lookback = min(3, abs(idx))
        if lookback > 0:
            fast_slope = (df['ema14'].iloc[idx] - df['ema14'].iloc[idx - lookback]) / lookback
            slow_slope = (df['ema55'].iloc[idx] - df['ema55'].iloc[idx - lookback]) / lookback
            slope_diff = (fast_slope - slow_slope) / df['close'].iloc[idx]
            return float(np.degrees(np.arctan(slope_diff * 100)))
        return 0.0
    
    def scan(self, symbols: List[str] = None, timeframes: List[str] = None):
        """Main scanning function"""
        if symbols is None:
            symbols = POPULAR_SYMBOLS
        
        if timeframes is None:
            timeframes = ['15m', '1h', '4h', '1d']
        
        print("\n🚀 Starting AMA Pro Scan (Manual Symbol List)")
        print("=" * 60)
        print(f"📊 Scanning {len(symbols)} symbols across {len(timeframes)} timeframes")
        print("⏱️  This will take a few minutes due to rate limiting...")
        print("=" * 60)
        
        all_signals = []
        total_scanned = 0
        
        # Scan each timeframe
        for tf in timeframes:
            print(f"\n⏰ Scanning timeframe: {tf}")
            tf_signals = []
            success_count = 0
            
            for i, symbol in enumerate(symbols, 1):
                print(f"  [{i}/{len(symbols)}] {symbol:15} ", end='', flush=True)
                
                # Fetch data
                df = self.fetch_ohlcv(symbol, tf)
                
                if df is not None:
                    # Calculate signals
                    signals = self.calculate_ama_signals(df, symbol, tf)
                    tf_signals.extend(signals)
                    success_count += 1
                    total_scanned += 1
                    
                    if signals:
                        print(f"✅ {len(signals)} signal(s)")
                    else:
                        print(f"⚪ No signals")
                else:
                    print(f"❌ Failed")
                
                # Rate limiting delay
                time.sleep(0.15)  # 150ms between requests
            
            print(f"\n  ✅ Completed {tf}: {len(tf_signals)} signals from {success_count}/{len(symbols)} symbols")
            all_signals.extend(tf_signals)
        
        print("\n" + "=" * 60)
        print(f"✅ Scan complete!")
        print(f"  • Symbols scanned: {total_scanned}/{len(symbols) * len(timeframes)}")
        print(f"  • Total signals: {len(all_signals)}")
        print("=" * 60)
        
        return all_signals
    
    def save_results(self, signals: List[dict], filename: str = None):
        """Save results to CSV"""
        if not signals:
            print("\n📧 No signals found")
            return
        
        df = pd.DataFrame(signals)
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"ama_scan_results_{timestamp}.csv"
        
        df.to_csv(filename, index=False)
        print(f"\n💾 Results saved to {filename}")
        
        # Print summary
        print("\n📊 Signal Summary:")
        print(f"  • Total signals: {len(signals)}")
        print(f"  • LONG signals: {len([s for s in signals if s['type'] == 'LONG'])}")
        print(f"  • SHORT signals: {len([s for s in signals if s['type'] == 'SHORT'])}")
        
        # Group by timeframe
        print("\n📈 Signals by Timeframe:")
        for tf in sorted(df['timeframe'].unique()):
            count = len(df[df['timeframe'] == tf])
            print(f"  • {tf:4}: {count} signals")
        
        # Show top signals
        df_sorted = df.sort_values('angle', key=abs, ascending=False)
        print("\n🔥 Top 15 Signals by Angle:")
        print("-" * 80)
        for idx, row in df_sorted.head(15).iterrows():
            print(f"  {row['symbol']:12} [{row['timeframe']:4}] {row['type']:5} | "
                  f"Angle: {row['angle']:7.2f}° | RSI: {row['rsi']:5.1f} | {row['time']}")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AMA Pro Scanner - Manual Symbol List')
    parser.add_argument('--timeframes', type=str, default='1h,4h,1d', 
                       help='Comma-separated timeframes (default: 1h,4h,1d)')
    parser.add_argument('--top', type=int, default=100,
                       help='Scan top N symbols from list (default: 100)')
    
    args = parser.parse_args()
    
    # Parse timeframes
    timeframes = [tf.strip() for tf in args.timeframes.split(',')]
    
    # Get symbol subset
    symbols = POPULAR_SYMBOLS[:args.top]
    
    print(f"\n⚙️  Configuration:")
    print(f"  • Symbols: {len(symbols)}")
    print(f"  • Timeframes: {', '.join(timeframes)}")
    print(f"  • Total requests: ~{len(symbols) * len(timeframes)}")
    print(f"  • Estimated time: ~{int(len(symbols) * len(timeframes) * 0.15 / 60)} minutes")
    
    # Create scanner
    scanner = AMAScanner()
    
    # Run scan
    signals = scanner.scan(symbols=symbols, timeframes=timeframes)
    
    # Save results
    scanner.save_results(signals)

if __name__ == '__main__':
    main()
