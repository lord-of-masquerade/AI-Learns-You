# AI Learns You

Adaptive study intelligence app built with Streamlit and scikit-learn.

## Features

1. Auto model improvement via retraining on user history.
2. ML-based study-hour recommendation model (no fixed thresholds).
3. PDF analyzer for topic complexity and effort multiplier.
4. PDF-to-quiz and question converter.
5. Spider charts for overall and per-subject performance profile.
6. Technique checker with method-fit scoring.
7. RL planner with persistent Q-table memory and adaptive slot planning.
8. Intelligence dashboard with trend comparison, heatmap, and consistency score.
9. Behavior pattern detection for best windows, draining subjects, and burnout signals.
10. Personal memory profile that adapts recommendations over time.

## Run

```bash
pip install -r requirements.txt
python src/train.py
streamlit run app.py
```

## Structure

- `app.py` - main Streamlit app.
- `src/intelligence.py` - training, analytics, PDF analysis/export, quiz generation, RL planner, and profile helpers.
- `src/train.py` - trains and saves productivity + recommendation models.
- `data/` - source data + user history/profile.
- `models/` - persisted model artifacts.
