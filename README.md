# AI Learns You

An intelligent, adaptive study companion that learns from your behavior patterns to optimize your learning efficiency. Built with machine learning and reinforcement learning, this application provides personalized study recommendations, complexity analysis, and performance tracking.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)

## 🌟 Features

### 🤖 Intelligent Prediction & Recommendation
- **ML-Based Productivity Prediction**: Random Forest models predict your productivity based on study hours, focus level, distractions, and sleep patterns
- **Adaptive Study Hour Recommendations**: No fixed thresholds—the system learns from your history to recommend optimal study durations
- **Auto-Retraining**: Models automatically improve by retraining on your accumulated user history

### 📚 PDF Analysis & Processing
- **Complexity Analyzer**: Automatically analyzes PDF documents for topic complexity and calculates effort multipliers
- **Text Extraction**: Robust PDF text extraction with encoding handling
- **Quiz Generation**: Converts PDF content into practice quizzes with questions and answers
- **PDF Export**: Export generated quizzes as downloadable PDF files

### 📊 Performance Analytics
- **Spider Charts**: Visualize overall and per-subject performance profiles
- **Intelligence Dashboard**: Track trends with comparison charts, heatmaps, and consistency scores
- **Behavior Pattern Detection**: 
  - Identifies your best study time windows
  - Detects subjects that drain your energy
  - Warns about potential burnout signals

### 🧠 Reinforcement Learning Planner
- **Persistent Q-Table Memory**: RL agent that learns optimal study schedules over time
- **Adaptive Slot Planning**: Dynamically plans study sessions based on learned patterns
- **State-Based Decision Making**: Uses subject, fatigue, and time-of-day to make intelligent scheduling decisions

### 🎯 Study Optimization Tools
- **Technique Checker**: Evaluates study methods with method-fit scoring
- **Focus Session Timer**: Pomodoro-style focused study sessions
- **Personal Memory Profile**: Adapts recommendations based on your unique learning patterns

### 🎨 Modern UI
- **Multiple Themes**: Neon Grid, Cyber Teal, and Sunset Pulse themes
- **Responsive Design**: Clean, modern interface with gradient panels and glass-morphism effects
- **Real-time Updates**: Live tracking of study sessions and performance metrics

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/AI-Learns-You.git
   cd AI-Learns-You
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train initial models**
   ```bash
   python src/train.py
   ```
   This will create the initial ML models in the `models/` directory.

4. **Launch the application**
   ```bash
   streamlit run app.py
   ```

5. **Access the app**
   Open your browser and navigate to `http://localhost:8501`

## 📁 Project Structure

```
AI-Learns-You/
├── app.py                          # Main Streamlit application
├── src/
│   ├── intelligence.py             # Core ML/RL logic and utilities
│   └── train.py                    # Model training script
├── data/
│   ├── study_data.csv              # Base training dataset
│   ├── user_history.csv            # User interaction history
│   ├── user_profile.json           # Personalized user profile
│   └── rl_memory.json              # RL Q-table persistence
├── models/
│   ├── model.pkl                   # Productivity prediction model
│   ├── columns.pkl                 # Feature columns for productivity model
│   ├── recommendation_model.pkl    # Study hour recommendation model
│   └── recommendation_columns.pkl  # Feature columns for recommendation model
├── requirements.txt                # Python dependencies
├── LICENSE                         # License file
└── README.md                       # This file
```

## 🎮 Usage Guide

### 1. Predict Tab
- Enter your study parameters (hours, focus level, distractions, sleep)
- Select a subject
- Get instant productivity predictions
- View personalized study hour recommendations
- Save sessions to build your learning history

### 2. History Tab
- View all your past study sessions
- Download history as CSV
- Track your progress over time

### 3. PDF Analyzer
- Upload any educational PDF
- Get automatic complexity analysis
- Receive effort multiplier recommendations
- Extract and review content

### 4. Quiz Converter
- Upload PDF study materials
- Generate practice questions automatically
- Export quizzes as PDF for offline use
- Perfect for exam preparation

### 5. Technique Checker
- Evaluate different study methods
- Get method-fit scores based on your learning style
- Optimize your study approach

### 6. RL Planner
- Let the AI plan your study schedule
- Learns from your preferences and performance
- Adapts to fatigue and time-of-day patterns
- Provides optimal subject sequencing

### 7. Focus Session
- Start timed study sessions
- Track active learning time
- Build focused work habits

### 8. Dashboard
- Comprehensive performance overview
- Spider charts for subject-wise analysis
- Behavior pattern insights
- Consistency tracking
- Trend analysis with heatmaps

## 🧪 How It Works

### Machine Learning Models

1. **Productivity Predictor**: Random Forest Regressor (300 estimators)
   - Features: hours_studied, focus_level, distractions, sleep_hours, subject (one-hot encoded)
   - Target: productivity score
   - Automatically retrains when user history grows

2. **Study Hour Recommender**: Random Forest Regressor (220 estimators)
   - Features: subject, focus_level, past_productivity
   - Target: optimal hours to study (clipped 1-10)
   - Adapts to individual learning patterns

### Reinforcement Learning

- **State Space**: (subject, fatigue_level, time_window)
- **Action Space**: Study subject selection
- **Reward Function**: Based on productivity and fatigue management
- **Learning**: Q-learning with persistent memory
- **Exploration**: Epsilon-greedy policy for balanced learning

### Adaptive Learning

The system continuously improves by:
- Recording every study session
- Retraining models on accumulated data
- Updating the RL Q-table based on outcomes
- Adjusting recommendations to individual patterns

## 📊 Data Schema

### study_data.csv / user_history.csv
```csv
hours_studied,focus_level,distractions,sleep_hours,subject,productivity,timestamp
2.5,8,3,7,DSA,72.5,2024-04-26 10:30:00
```

### user_profile.json
```json
{
  "total_sessions": 25,
  "best_subject": "DSA",
  "avg_focus": 7.8,
  "preferred_time": "morning"
}
```

### rl_memory.json
```json
{
  "q_table": {...},
  "visits": {...},
  "last_update": "2024-04-26T12:00:00"
}
```

## 🎨 Customization

### Themes
Choose from three built-in themes in the sidebar:
- **Neon Grid**: Purple and teal cyber aesthetic
- **Cyber Teal**: Cool teal and aqua vibes
- **Sunset Pulse**: Warm sunset color palette

### Subjects
Default subjects can be modified in `src/intelligence.py`:
```python
DEFAULT_SUBJECTS = ["DSA", "OOP", "Maths", "Physics", "History"]
```

## 🔧 Configuration

### Model Parameters
Edit training parameters in `src/intelligence.py`:
```python
# Productivity model
RandomForestRegressor(n_estimators=300, random_state=42)

# Recommendation model
RandomForestRegressor(n_estimators=220, random_state=42)
```

### RL Parameters
Adjust learning parameters in `src/intelligence.py`:
```python
alpha = 0.1      # Learning rate
gamma = 0.9      # Discount factor
epsilon = 0.15   # Exploration rate
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/) for the web interface
- Machine learning powered by [scikit-learn](https://scikit-learn.org/)
- PDF processing with [PyPDF2](https://pypdf2.readthedocs.io/)
- Data analysis with [pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/)
- Visualizations with [Matplotlib](https://matplotlib.org/)

## 📧 Contact

For questions, feedback, or support, please open an issue on GitHub.

## 🗺️ Roadmap

Future enhancements planned:
- [ ] Mobile app version
- [ ] Multi-user support with cloud sync
- [ ] Integration with calendar apps
- [ ] Voice-activated study sessions
- [ ] Advanced analytics with deep learning
- [ ] Study group collaboration features
- [ ] Spaced repetition integration
- [ ] Export reports and certificates

---

**Made with ❤️ for students who want to study smarter, not harder**