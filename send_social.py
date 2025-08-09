#!/usr/bin/env python3
import requests, urllib.parse as u, json

BOT = "8378113775:AAFtZ6Swh8DuRJQt9lAW19aF7-OQg7BMRtg"
CHAT = "6839057822"

buy_price = 19.80
take_profit = 21.14  # 2:1 RR ratio
stop_loss = 19.13

# ---- CALCULATIONS ----
profit_per_unit = take_profit - buy_price
loss_per_unit = buy_price - stop_loss
rr_ratio = round(profit_per_unit / loss_per_unit, 2)
profit_pct = round((profit_per_unit / buy_price) * 100, 2)
loss_pct = round((loss_per_unit / buy_price) * 100, 2)

# Benchmark comparisons (typical daily returns)
spy_daily = 0.04  # SPY average daily return
btc_daily = 0.79  # BTC average daily return

# Trade quality
trade_quality = "🔥 EXCELLENT" if rr_ratio >= 3 else "✅ GOOD" if rr_ratio >= 2 else "⚠️ RISKY"

# ---- CHART URLs ----
def qc(params): return "https://quickchart.io/chart?" + u.urlencode({"c": json.dumps(params), "width": 800, "height": 400, "backgroundColor": "white"})

# Simple risk vs reward chart
risk_reward_chart = qc({
  "type": "horizontalBar",
  "data": {
    "labels": ["PROFIT TARGET", "RISK", "SPY100 DAILY", "BTC DAILY"],
    "datasets": [{
      "label": "Performance %",
      "data": [profit_pct, -loss_pct, spy_daily, btc_daily],
      "backgroundColor": [
        "rgba(34,197,94,0.9)",   # Green for profit
        "rgba(239,68,68,0.9)",   # Red for risk
        "rgba(59,130,246,0.7)",  # Blue for SPY
        "rgba(251,146,60,0.7)"   # Orange for BTC
      ],
      "borderWidth": 0
    }]
  },
  "options": {
    "scales": {
      "x": {
        "beginAtZero": True,
        "min": -4,
        "max": 8,
        "grid": {
          "color": "rgba(200,200,200,0.3)"
        },
        "ticks": {
          "stepSize": 2,
          "font": {
            "size": 14
          }
        }
      },
      "y": {
        "grid": {
          "display": False
        },
        "ticks": {
          "font": {
            "size": 16,
            "weight": "500"
          },
          "color": "#666"
        }
      }
    },
    "plugins": {
      "legend": {
        "display": True,
        "position": "top",
        "labels": {
          "font": {
            "size": 16
          }
        }
      },
      "title": {
        "display": False
      }
    },
    "layout": {
      "padding": 20
    },
    "maintainAspectRatio": False,
    "responsive": True
  }
})

API = lambda m: f"https://api.telegram.org/bot{BOT}/{m}"

# ---- MAIN ALERT ----
r1 = requests.post(API("sendMessage"),json={
  "chat_id":CHAT,
  "parse_mode":"Markdown",
  "disable_web_page_preview":True,
  "text":(
    f"🎯 *LINK/USDT* · RR {rr_ratio}:1 {trade_quality}\n"
    f"━━━━━━━━━━━━━━━\n"
    f"📍 Entry: ${buy_price:.2f}\n"
    f"🎯 Target: ${take_profit:.2f} (+{profit_pct}%)\n"
    f"🛡️ Stop: ${stop_loss:.2f} (-{loss_pct}%)\n\n"
    f"Risk $1 to make $2\n\n"
    f"ℹ️ *What is LINK/USDT?*\n"
    f"LINK powers Chainlink - the bridge connecting banks & blockchains to real-world data. "
    f"Trading vs USDT (dollar stablecoin).\n\n"
    f"💎 *TRADE NOW:*\n"
    f"[Join Binance & Get Rewards →](https://www.binance.com/activity/referral-entry/CPA?ref=CPA_00597U293M)\n"
  )
},timeout=15)
print(f"Message: {r1.status_code}")

# ---- CHART ----
r2 = requests.post(API("sendPhoto"),json={
  "chat_id":CHAT,
  "photo":risk_reward_chart,
  "caption":f"📊 Trade: +{profit_pct}% vs Risk: -{loss_pct}% | SPY100: +{spy_daily}% | BTC: +{btc_daily}%"
},timeout=15)
print(f"Chart: {r2.status_code}")