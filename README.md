# AI Learns You

Adaptive study intelligence app built with Streamlit and scikit-learn.

## Features

1. Auto model improvement via retraining on user history.
2. ML-based study-hour recommendation model (no fixed thresholds).
3. PDF analyzer for topic complexity and effort multiplier.
4. Voice input parsing for natural-language study logs.
5. Intelligence dashboard with trend comparison, heatmap, and consistency score.
6. Behavior pattern detection for best windows, draining subjects, and burnout signals.
7. Personal memory profile that adapts recommendations over time.

## Run

```bash
pip install -r requirements.txt
python src/train.py
streamlit run app.py
```

## Structure

- `app.py` - main Streamlit app.
- `src/intelligence.py` - training, analytics, PDF, voice parsing, and profile helpers.
- `src/train.py` - trains and saves productivity + recommendation models.
- `data/` - source data + user history/profile.
- `models/` - persisted model artifacts.
