#!/usr/bin/env python3
"""
Quick test script to check Binance API connectivity
"""

import asyncio
import aiohttp
import json

async def test_binance_connection():
    """Test connection to Binance API"""
    
    print("🔍 Testing Binance Futures API connection...")
    
    url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
    
    try:
        async with aiohttp.ClientSession() as session:
            print(f"📡 Fetching from: {url}")
            
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                print(f"✅ Response status: {response.status}")
                
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Received {len(data)} trading pairs")
                    
                    # Filter USDT pairs
                    usdt_pairs = [item for item in data if 'USDT' in item['symbol']]
                    print(f"✅ Found {len(usdt_pairs)} USDT pairs")
                    
                    # Sort by volume
                    sorted_pairs = sorted(usdt_pairs, key=lambda x: float(x['quoteVolume']), reverse=True)
                    
                    # Show top 10
                    print("\n📊 Top 10 by volume:")
                    for i, pair in enumerate(sorted_pairs[:10], 1):
                        symbol = pair['symbol']
                        volume = float(pair['quoteVolume'])
                        price_change = float(pair['priceChangePercent'])
                        print(f"  {i}. {symbol:12} | Volume: ${volume:,.0f} | Change: {price_change:+.2f}%")
                    
                    return sorted_pairs[:100]
                else:
                    print(f"❌ Error: HTTP {response.status}")
                    text = await response.text()
                    print(f"Response: {text[:500]}")
                    return None
                    
    except asyncio.TimeoutError:
        print("❌ Connection timeout - check your internet connection")
        return None
    except aiohttp.ClientError as e:
        print(f"❌ Connection error: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None

async def test_klines():
    """Test fetching candlestick data"""
    print("\n\n🔍 Testing OHLCV data fetch...")
    
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {
        'symbol': 'BTCUSDT',
        'interval': '1h',
        'limit': 100
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Received {len(data)} candles for BTCUSDT 1h")
                    
                    # Show last candle
                    last = data[-1]
                    print(f"\n📊 Latest candle:")
                    print(f"  Time: {last[0]}")
                    print(f"  Open: {last[1]}")
                    print(f"  High: {last[2]}")
                    print(f"  Low: {last[3]}")
                    print(f"  Close: {last[4]}")
                    print(f"  Volume: {last[5]}")
                    
                    return True
                else:
                    print(f"❌ Error: HTTP {response.status}")
                    return False
                    
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Binance API Connection Test")
    print("=" * 60)
    
    asyncio.run(test_binance_connection())
    asyncio.run(test_klines())
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
