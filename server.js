// server.js — Local proxy for AI That Learns You
// Forwards requests to Anthropic API so CORS is bypassed locally
//
// Setup:
//   1. npm install express cors node-fetch
//   2. set ANTHROPIC_API_KEY=sk-ant-...
//   3. node server.js
//   4. Open http://localhost:3000 in browser

const express = require('express');
const cors    = require('cors');
const app     = express();

app.use(cors());
app.use(express.json());
app.use(express.static('.'));   // serves demo.html from same folder

// Proxy endpoint — demo.html calls /api/claude instead of anthropic.com directly
app.post('/api/claude', async (req, res) => {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    return res.status(500).json({ error: 'ANTHROPIC_API_KEY not set. Run: set ANTHROPIC_API_KEY=sk-ant-...' });
  }

  try {
    const fetch = (await import('node-fetch')).default;
    const response = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type':         'application/json',
        'x-api-key':            apiKey,
        'anthropic-version':    '2023-06-01',
      },
      body: JSON.stringify(req.body),
    });

    const data = await response.json();
    res.json(data);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

const PORT = 3000;
app.listen(PORT, () => {
  console.log(`\n✅ Server running at http://localhost:${PORT}`);
  console.log(`   Open http://localhost:${PORT}/demo.html\n`);
});
