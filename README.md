# Ligue 1 Match Prediction

A machine learning model to predict Ligue 1 football match outcomes using EWMA analysis and Monte Carlo simulation.

## What It Does

The project analyzes match data to predict probabilities for three outcomes: home win, draw, or away win. It also calculates fair odds based on the predicted probabilities.

**How it works:**
- Loads historical match data (goals scored/conceded)
- Calculates team strength using EWMA (Exponentially Weighted Moving Average) with span=10
- Separates home and away statistics for each team
- Uses Monte Carlo simulation with Poisson distribution to predict outcomes

## Installation

### Requirements
- Python 3.9+
- pip or uv

### Setup

```bash
# Clone or download the project
cd predict_ligue1

# Install dependencies
pip install -r requirements.txt
# OR with uv:
uv sync
```

## Usage

### Option 1: Web Interface (Recommended)

```bash
python app.py
```

Then open http://localhost:5000 in your browser.

- Select home and away teams from dropdowns
- Click "Predict Match"
- View probabilities and fair odds

### Option 2: Command Line

```bash
python main.py
```

Edit the team names inside `main.py` to change which match to predict.

## Project Structure

```
predict_ligue1/
├── app.py                # Web app with full interface
├── main.py               # CLI script for quick predictions
├── requirements.txt      # Python dependencies
├── Dockerfile            # For Hugging Face Spaces deployment
├── pyproject.toml        # Project config
└── README.md             # This file
```

## Features

- **Real-time data**: Uses latest Ligue 1 2025/2026 season data
- **Recent form focus**: EWMA(10) emphasizes recent matches
- **Home/Away split**: Separates statistics by match location
- **Fair odds**: Converts probabilities to betting odds
- **Clean UI**: Simple, professional web interface

## Model Details

| Parameter | Value |
|-----------|-------|
| EWMA Span | 10 |
| Simulations | 10,000 |
| Distribution | Poisson |
| Data Source | football-data.co.uk |
| Goals Cap | 3.5 (to reduce outliers) |

## Deployment

### Hugging Face Spaces

1. Create a new Space on Hugging Face
2. Upload `app.py`, `requirements.txt`, and `Dockerfile`
3. Space automatically picks up the Dockerfile and deploys
4. Your app will be live at `huggingface.co/spaces/[username]/[space-name]`

### Local Server

For production on your own server:
```bash
gunicorn app:app --bind 0.0.0.0:5000
```

## Files Explained

- **app.py**: Complete web application with built-in model. No external dependencies except Flask/pandas/numpy
- **main.py**: Simpler CLI version for quick terminal predictions
- **requirements.txt**: All Python packages needed
- **Dockerfile**: Container setup for cloud deployment

## Example Output

```
Expected Goals:
  Home Team: 1.85 goals
  Away Team: 0.92 goals

Probabilities (10,000 simulations):
  Home Win: 43.1%
  Draw:     20.7%
  Away Win: 36.2%

Fair Odds:
  Home Win: 2.32
  Draw:     4.83
  Away Win: 2.76
```

## Notes

- Model uses real Ligue 1 data from 2025/2026 season
- Predictions are probabilistic estimates, not guarantees
- Recent form is weighted more heavily (EWMA effect)
- Works best with teams that have played 5+ matches
- Goals are capped at 3.5 to reduce impact of blowouts

## License

Open source - feel free to use and modify

## Support

For issues or questions, check the code comments or open an issue in your repository.
