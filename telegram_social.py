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

# Benchmark comparisons (typical daily returns and volatility)
spy_daily = 0.04  # SPY average daily return
btc_daily = 0.79  # BTC average daily return
spy_risk = 1.2   # SPY typical daily volatility
btc_risk = 3.5   # BTC typical daily volatility

# Calculate RR ratios for comparison
spy_rr = round(spy_daily / spy_risk, 2) if spy_risk > 0 else 0
btc_rr = round(btc_daily / btc_risk, 2) if btc_risk > 0 else 0

# Trade quality
trade_quality = "🔥 EXCELLENT" if rr_ratio >= 3 else "✅ GOOD" if rr_ratio >= 2 else "⚠️ RISKY"

# ---- CHART URLs ----
def qc(params): return "https://quickchart.io/chart?" + u.urlencode({"c": json.dumps(params), "width": 800, "height": 350, "backgroundColor": "white"})

# Calculate RR ratios for display
spy_rr_display = f"{spy_rr}:1" if spy_rr > 0 else "Poor"
btc_rr_display = f"{btc_rr}:1" if btc_rr > 0 else "Poor"
trade_rr_display = f"{rr_ratio}:1"

# Simple risk vs reward chart
risk_reward_chart = qc({
  "type": "horizontalBar",
  "data": {
    "labels": [
      "Current Strategy",
      "SPY100",
      "BTC Holding"
    ],
    "datasets": [
      {
        "label": "RISK",
        "data": [-loss_pct, -spy_risk, -btc_risk],
        "backgroundColor": "#DC2626",
        "borderWidth": 0
      },
      {
        "label": "REWARD",
        "data": [profit_pct, spy_daily, btc_daily],
        "backgroundColor": "#16A34A",
        "borderWidth": 0
      }
    ]
  },
  "options": {
    "indexAxis": "y",
    "scales": {
      "x": {
        "stacked": True,
        "beginAtZero": True,
        "min": -4,
        "max": 8,
        "grid": {
          "color": "rgba(200,200,200,0.3)",
          "drawBorder": False
        },
        "ticks": {
          "stepSize": 2,
          "font": {
            "size": 14
          },
          "callback": "function(value) { return Math.abs(value) + '%'; }"
        }
      },
      "y": {
        "stacked": True,
        "grid": {
          "display": False
        },
        "ticks": {
          "font": {
            "size": 16,
            "weight": "600"
          },
          "color": "#333"
        }
      }
    },
    "plugins": {
      "legend": {
        "display": True,
        "position": "top",
        "labels": {
          "font": {
            "size": 14,
            "weight": "bold"
          },
          "usePointStyle": False,
          "padding": 20,
          "generateLabels": "function(chart) { return chart.data.datasets.map(function(dataset, i) { return { text: dataset.label, fillStyle: dataset.backgroundColor, hidden: false, index: i }; }); }"
        }
      },
      "title": {
        "display": True,
        "text": "Risk-Reward Ratios in Trading",
        "font": {
          "size": 18,
          "weight": "bold"
        },
        "padding": 20
      },
      "tooltip": {
        "callbacks": {
          "label": "function(context) { return context.dataset.label + ': ' + Math.abs(context.parsed.x).toFixed(2) + '%'; }"
        }
      },
      "datalabels": {
        "display": True,
        "align": "right",
        "anchor": "end",
        "formatter": "function(value, context) { if (context.datasetIndex === 1 && context.dataIndex === 0) return '2:1'; if (context.datasetIndex === 1 && context.dataIndex === 1) return '0.03:1'; if (context.datasetIndex === 1 && context.dataIndex === 2) return '0.23:1'; return ''; }",
        "font": {
          "size": 14,
          "weight": "bold"
        },
        "color": "#333"
      }
    },
    "layout": {
      "padding": 30
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
  "caption":f"📊 Trade smarter with {rr_ratio}:1 risk-reward ratio"
},timeout=15)
print(f"Chart: {r2.status_code}")