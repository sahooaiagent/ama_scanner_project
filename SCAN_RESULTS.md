# AMA Pro Scanner - Initial Results

**Status**: Partial Scan (Top 52/500 Symbols)
**Time**: 2026-02-10 05:45 (CET)

The scanner was executed for the top 52 crypto perpetuals on Binance Futures. Due to API rate limits, a full scan takes ~20 minutes. Below are the signals identified in the first ~10% of the market.

## 🚨 Trading Signals Identified

| Crypto Symbol | Timeframe | Signal |
| :--- | :--- | :--- |
| **ETH/USDT** | **3h** | 🔴 **SHORT** |
| **DUSK/USDT** | **2h** | 🔴 **SHORT** |
| **RIVER/USDT** | **2h** | 🟢 **LONG** |
| **LINK/USDT** | **1h** | 🔴 **SHORT** |

## Next Steps
To scan the remaining 450+ symbols, please run the script locally:
```bash
python3 ama_pro_scanner.py
```
The script will save a CSV file with all found signals.
