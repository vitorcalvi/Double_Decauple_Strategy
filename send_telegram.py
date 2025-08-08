# trader_send_test.py
import requests, time, urllib.parse as u

BOT="8378113775:AAFtZ6Swh8DuRJQt9lAW19aF7-OQg7BMRtg"
CHAT="6839057822"
DASH_URL="https://your-dashboard.example/trades"  # ← replace
SYMBOL="ETHUSDT"
API=lambda m:f"https://api.telegram.org/bot{BOT}/{m}"
POST=lambda m,p: requests.post(API(m),json=p,timeout=15).ok

# Simple sparkline & bar chart via QuickChart (replace data as needed)
def qc(params): return "https://quickchart.io/chart?"+u.urlencode({"c":params})
pnl_line=qc("""{
  type:'line',data:{labels:[1,2,3,4,5,6,7,8],
  datasets:[{data:[-12,-7,-3,4,9,6,11,15],fill:false}]},
  options:{legend:{display:false},scales:{xAxes:[{display:false}],yAxes:[{display:false}]}}
}""")
fee_bar=qc("""{
  type:'bar',data:{labels:['Fees','Slippage','Funding','Net%'],
  datasets:[{data:[0.38,0.22,0.09,1.15]}]},
  options:{legend:{display:false},scales:{xAxes:[{display:false}],yAxes:[{display:false}]}}
}""")

# 0) Typing…
POST("sendChatAction",{"chat_id":CHAT,"action":"typing"})

# 1) Trade ALERT with inline actions
POST("sendMessage",{
  "chat_id":CHAT,"parse_mode":"HTML","disable_web_page_preview":True,
  "text":(
    f"🚨 <b>ALERT</b> · {SYMBOL} · <b>Breakout LONG</b>\n"
    "Price: <b>3,271.50</b> | Vol↑\n"
    "TP: 3,315 · SL: 3,229 · RR: 1.1\n"
    "<i>Reason:</i> 20m BO + ADX 27 + MFI 62\n"
    "<code>entry=3271.5 size=3.0 risk=0.65% fee≈0.055%</code>"
  ),
  "reply_markup":{
    "inline_keyboard":[
      [{"text":"✅ Acknowledge","callback_data":"ack"}],
      [{"text":"🟢 Close @ Market","callback_data":"close_mkt"},
       {"text":"🟡 SL → BE","callback_data":"sl_to_be"}],
      [{"text":"📊 Open Dashboard","url":DASH_URL}]
    ]
  }
})

# 2) Risk/PnL block (HTML)
POST("sendMessage",{
  "chat_id":CHAT,"parse_mode":"HTML",
  "text":(
    "🧮 <b>Risk Snapshot</b>\n"
    "<pre>"
    "Acct: $25,000    Risk/trade: $162 (0.65%)\n"
    "Max DD (7d): -3.2%   Vol regime: MED\n"
    "Fees (30d): 0.38% of notional   Taker%: 57%\n"
    "Funding (avg 8h): 2.6 bps\n"
    "</pre>"
  )
})

# 3) Fixed-width table (MarkdownV2) — good for fee/funding detail
POST("sendMessage",{
  "chat_id":CHAT,"parse_mode":"MarkdownV2",
  "text":(
    "*Fees / Funding / Slip*\n"
    "```\n"
    "Metric        Value\n"
    "------------  -----\n"
    "Maker fee     0.00%\n"
    "Taker fee     0.055%\n"
    "Slip (avg)    0.022%\n"
    "Funding (8h)  0.030%\n"
    "Breakeven     0.19%\n"
    "```\n"
    "_Note: breakeven ≈ 2×(fees+slip) + funding_\n"
  )
})

# 4) PnL sparkline
POST("sendPhoto",{
  "chat_id":CHAT,"photo":pnl_line,
  "caption":"📈 PnL sparkline (last 8 trades)"
})

# 5) Cost components mini chart
POST("sendPhoto",{
  "chat_id":CHAT,"photo":fee_bar,
  "caption":"💸 Cost load: Fees/Slip/Funding vs Net%"
})

# 6) Order confirm mock (MarkdownV2) + quick reply keyboard to simulate choices
POST("sendMessage",{
  "chat_id":CHAT,"parse_mode":"MarkdownV2",
  "text":"*Order Placed*  `ETHUSDT`  `BUY 3.0`  @ `3271\\.5`  SL `3229`  TP `3315`"
})
POST("sendMessage",{
  "chat_id":CHAT,"text":"Quick actions ⌨️",
  "reply_markup":{"keyboard":[[{"text":"Move SL→BE"},{"text":"Close 50%"}],[{"text":"Cancel TP"}]],"resize_keyboard":True,"one_time_keyboard":True}
})

# 7) Cooldown/funding/spread notices
POST("sendMessage",{
  "chat_id":CHAT,"parse_mode":"HTML",
  "text":"⏱ <b>Cooldown</b>: 45s remaining · 🧲 <b>Funding</b>: 2.8 bps · ↔️ <b>Spread</b>: 0.06%"
})

# 8) Remove keyboard + done
time.sleep(1)
POST("sendMessage",{"chat_id":CHAT,"text":"Removing keyboard…","reply_markup":{"remove_keyboard":True}})
POST("sendMessage",{"chat_id":CHAT,"parse_mode":"HTML","text":"<b>✅ Trader test complete.</b> Check alert card, actions, PnL block, fee table, and charts."})
