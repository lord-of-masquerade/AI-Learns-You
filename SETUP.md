# Running AI That Learns You Locally

## Why the demo looks different / Claude AI features don't work

When you **double-click demo.html** and open it as `file://`, two things break:

| Problem | Cause | Fix |
|---|---|---|
| Fonts may not load | Google Fonts needs internet | Need internet connection |
| Claude AI tabs do nothing | Browser blocks direct calls to api.anthropic.com (CORS) | Use the local proxy (below) |

---

## Step-by-step local setup

### Option A — Node.js (recommended)

**Step 1 — Check Node.js is installed**
```
node --version
```
If not installed → https://nodejs.org (download LTS)

**Step 2 — Install dependencies**
```
npm install express cors node-fetch
```

**Step 3 — Get your Anthropic API key**
Go to https://console.anthropic.com → API Keys → Create Key

**Step 4 — Set the API key**

Mac / Linux:
```
export ANTHROPIC_API_KEY=sk-ant-api03-YOUR_KEY_HERE
```

Windows CMD:
```
set ANTHROPIC_API_KEY=sk-ant-api03-YOUR_KEY_HERE
```

Windows PowerShell:
```
$env:ANTHROPIC_API_KEY="sk-ant-api03-YOUR_KEY_HERE"
```

**Step 5 — Start the proxy**
```
node server.js
```

**Step 6 — Open in browser**
```
http://localhost:3000/demo.html
```

✅ All Claude AI features (PDF Analyser, MCQ Quiz, Technique Checker, RL Planner notes) will now work.

---

### Option B — Python

**Step 1 — Install dependencies**
```
pip install flask flask-cors requests
```

**Step 2 — Set API key** (same as Step 3-4 above)

**Step 3 — Start the proxy**
```
python server.py
```

**Step 4 — Open in browser**
```
http://localhost:3000/demo.html
```

---

## File placement

Put these files in the **same folder**:

```
AI-Learns-You/          ← your project root
├── demo.html           ← the dashboard
├── server.js           ← Node proxy  (Option A)
├── server.py           ← Python proxy (Option B)
└── SETUP.md            ← this file
```

---

## Productivity Predictor (no API needed)

The **⚡ Predict** tab uses local JS logic — it works even without a proxy.  
Only the AI-powered tabs need the proxy running.
