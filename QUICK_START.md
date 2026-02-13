# 🚀 Quick Start Guide - AMA Pro Scanner

## Starting the Dashboard (Single Command)

Simply run:

```bash
python3 start_dashboard.py
```

This will:
✅ Start the backend server automatically
✅ Open the dashboard in your browser
✅ Handle all setup for you

**To stop**: Press `Ctrl+C` in the terminal

---

## Alternative: Manual Start

If you prefer to run components separately:

### Terminal 1 - Backend Server
```bash
cd web_dashboard/backend
python3 main.py
```

### Terminal 2 - Frontend
```bash
# Open the HTML file in your browser
open web_dashboard/frontend/index.html
```

---

## Running a Scan

### Option 1: Use the Web Dashboard
1. Select timeframes (e.g., 1h, 4h, 1d)
2. Choose number of cryptos to scan (20-500)
3. Click "Run Scanner Now"

### Option 2: Command Line
```bash
# Scan top 100 symbols with default timeframes
python3 ama_pro_scanner.py 100

# Scan with custom timeframes
python3 ama_pro_scanner.py 100 --timeframes 1h,4h,1d
```

---

## Features at a Glance

### New in this version:
- 🔍 **Real-time Search** - Filter signals by symbol name
- 🎯 **Smart Filters** - Filter by signal type (LONG/SHORT) and timeframe
- 📊 **Sortable Columns** - Click any column header to sort
- 💾 **CSV Export** - Download results with one click
- 🎨 **Toast Notifications** - Beautiful feedback for all actions
- ✅ **Improved Signal Validation** - Only shows signals that haven't been invalidated
- 🚀 **Performance Optimized** - Faster scanning with reduced API calls
- 📱 **Mobile Responsive** - Works great on all devices

---

## Signal Validation Logic

Signals are only shown when:

### LONG Signals
- ✅ Buy candle occurred in the **last 5 closed candles**
- ✅ Price has **NOT crossed above** the buy candle's high
- ✅ Passes regime and trend filters

### SHORT Signals
- ✅ Sell candle occurred in the **last 5 closed candles**
- ✅ Price has **NOT crossed below** the sell candle's low
- ✅ Passes regime and trend filters

This ensures you only see fresh, valid signals that haven't played out yet!

---

## Troubleshooting

### Port 8000 already in use
The launcher will detect this and just open the browser. If you need to restart:
```bash
# Find and kill the process
lsof -ti:8000 | xargs kill -9

# Then run the launcher again
python3 start_dashboard.py
```

### No signals found
This is normal! The scanner uses strict filters to ensure quality signals.
- Try scanning more cryptos (e.g., 200-500)
- Try different timeframes
- Check the logs in the "Diagnostics" section

---

## Need Help?

- Check the **Diagnostics** section in the dashboard for live logs
- Review scan parameters in the **Configuration** section
- Email notifications are sent automatically with scan results

**Happy Trading! 🎯📈**
