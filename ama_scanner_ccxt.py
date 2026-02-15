#!/usr/bin/env python3
"""
AMA Pro Scanner - CCXT Edition
Uses CCXT library which handles rate limiting automatically
"""

import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime
from typing import List, Optional

class AMAScanner:
    def __init__(self):
        # Initialize Binance with rate limiting
        self.exchange = ccxt.binance({
            'enableRateLimit': True,  # CCXT handles rate limiting
            'options': {
                'defaultType': 'future',  # Use futures
            }
        })
        
    def get_top_symbols(self, limit: int = 100) -> List[str]:
        """Get top trading symbols by volume"""
        print("🔍 Fetching top symbols...")
        
        try:
            # Fetch all tickers
            tickers = self.exchange.fetch_tickers()
            
            # Filter USDT pairs and sort by volume
            usdt_pairs = {
                symbol: ticker for symbol, ticker in tickers.items() 
                if '/USDT' in symbol
            }
            
            # Sort by quote volume
            sorted_pairs = sorted(
                usdt_pairs.items(),
                key=lambda x: float(x[1].get('quoteVolume', 0)),
                reverse=True
            )
            
            symbols = [pair[0] for pair in sorted_pairs[:limit]]
            print(f"✅ Found {len(symbols)} symbols")
            
            return symbols
            
        except Exception as e:
            print(f"❌ Error fetching symbols: {e}")
            return []
    
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
            # print(f"⚠️ Error fetching {symbol}: {e}")
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
        
        # Check for crossovers in last few candles
        for i in range(-5, 0):
            try:
                # Fast MA crossover slow MA
                fast_ma_current = df['ema14'].iloc[i]
                fast_ma_prev = df['ema14'].iloc[i-1]
                slow_ma_current = df['ema55'].iloc[i]
                slow_ma_prev = df['ema55'].iloc[i-1]
                
                # Long signal: fast crosses above slow
                if fast_ma_prev <= slow_ma_prev and fast_ma_current > slow_ma_current:
                    angle = self.calculate_angle(df, i)
                    if abs(angle) > 5:  # Filter weak signals
                        signals.append({
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'type': 'LONG',
                            'angle': angle,
                            'price': df['close'].iloc[i],
                            'time': df.index[i]
                        })
                
                # Short signal: fast crosses below slow
                elif fast_ma_prev >= slow_ma_prev and fast_ma_current < slow_ma_current:
                    angle = self.calculate_angle(df, i)
                    if abs(angle) > 5:  # Filter weak signals
                        signals.append({
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'type': 'SHORT',
                            'angle': angle,
                            'price': df['close'].iloc[i],
                            'time': df.index[i]
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
    
    def scan(self, symbol_limit: int = 100, timeframes: List[str] = None):
        """Main scanning function"""
        if timeframes is None:
            timeframes = ['15m', '30m', '1h', '4h', '1d']
        
        print("\n🚀 Starting AMA Pro Scan (CCXT Edition)")
        print("=" * 60)
        
        # Get top symbols
        symbols = self.get_top_symbols(symbol_limit)
        if not symbols:
            print("❌ No symbols found")
            return []
        
        all_signals = []
        
        # Scan each timeframe
        for tf in timeframes:
            print(f"\n⏰ Scanning timeframe: {tf}")
            tf_signals = []
            
            for i, symbol in enumerate(symbols, 1):
                print(f"  [{i}/{len(symbols)}] {symbol}...", end='\r')
                
                # Fetch data
                df = self.fetch_ohlcv(symbol, tf)
                
                # Calculate signals
                signals = self.calculate_ama_signals(df, symbol, tf)
                tf_signals.extend(signals)
                
                # CCXT handles rate limiting, but add small delay
                time.sleep(0.1)
            
            print(f"  ✅ Found {len(tf_signals)} signals in {tf}            ")
            all_signals.extend(tf_signals)
        
        print("\n" + "=" * 60)
        print(f"✅ Scan complete! Total signals: {len(all_signals)}")
        
        return all_signals
    
    def save_results(self, signals: List[dict], filename: str = None):
        """Save results to CSV"""
        if not signals:
            print("📧 No signals to save")
            return
        
        df = pd.DataFrame(signals)
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"ama_scan_results_{timestamp}.csv"
        
        df.to_csv(filename, index=False)
        print(f"💾 Results saved to {filename}")
        
        # Print summary
        print("\n📊 Signal Summary:")
        print(f"  • Total signals: {len(signals)}")
        print(f"  • LONG signals: {len([s for s in signals if s['type'] == 'LONG'])}")
        print(f"  • SHORT signals: {len([s for s in signals if s['type'] == 'SHORT'])}")
        
        # Show top signals
        df_sorted = df.sort_values('angle', key=abs, ascending=False)
        print("\n🔥 Top 10 Signals by Angle:")
        for idx, row in df_sorted.head(10).iterrows():
            print(f"  • {row['symbol']:12} [{row['timeframe']:4}] {row['type']:5} | Angle: {row['angle']:6.1f}°")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AMA Pro Scanner - CCXT Edition')
    parser.add_argument('--symbols', type=int, default=100, help='Number of symbols to scan')
    parser.add_argument('--timeframes', type=str, default='15m,1h,4h,1d', 
                       help='Comma-separated timeframes')
    
    args = parser.parse_args()
    
    # Parse timeframes
    timeframes = [tf.strip() for tf in args.timeframes.split(',')]
    
    # Create scanner
    scanner = AMAScanner()
    
    # Run scan
    signals = scanner.scan(symbol_limit=args.symbols, timeframes=timeframes)
    
    # Save results
    scanner.save_results(signals)

if __name__ == '__main__':
    main()
