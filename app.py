from flask import Flask, render_template_string, request, jsonify
import pandas as pd
import numpy as np

app = Flask(__name__)

# --- CHARGEMENT DES DONNÉES ---
def charger_donnees():
    df = pd.read_csv('https://www.football-data.co.uk/mmz4281/2526/F1.csv')
    df = df[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']]
    df.columns = ['home_team', 'away_team', 'home_goals', 'away_goals']
    
    df['home_goals_adj'] = df['home_goals'].clip(upper=3.5) 
    df['away_goals_adj'] = df['away_goals'].clip(upper=3.5)
    df['match_order'] = range(len(df))
    
    return df

def entrainer_modele(df, span=10):
    """Entraîne le modèle avec EWMA"""
    avg_h = df['home_goals_adj'].mean()
    avg_a = df['away_goals_adj'].mean()
    
    stats_globales = pd.DataFrame()
    
    # DOMICILE
    for team in df['home_team'].unique():
        mask = df['home_team'] == team
        home_data = df[mask].copy().sort_values('match_order')
        
        if len(home_data) > 0:
            att_ewma = home_data['home_goals_adj'].ewm(span=span).mean().iloc[-1]
            def_ewma = home_data['away_goals_adj'].ewm(span=span).mean().iloc[-1]
            
            stats_globales.loc[team, 'attaque_domicile'] = att_ewma
            stats_globales.loc[team, 'defense_domicile'] = def_ewma
    
    # EXTÉRIEUR
    for team in df['away_team'].unique():
        mask = df['away_team'] == team
        away_data = df[mask].copy().sort_values('match_order')
        
        if len(away_data) > 0:
            att_ewma = away_data['away_goals_adj'].ewm(span=span).mean().iloc[-1]
            def_ewma = away_data['home_goals_adj'].ewm(span=span).mean().iloc[-1]
            
            stats_globales.loc[team, 'attaque_exterieur'] = att_ewma
            stats_globales.loc[team, 'defense_exterieur'] = def_ewma
    
    stats_globales = stats_globales.fillna(avg_h)
    
    stats_globales['force_att_domicile'] = stats_globales['attaque_domicile'] / avg_h
    stats_globales['force_att_exterieur'] = stats_globales['attaque_exterieur'] / avg_a
    stats_globales['faibl_def_domicile'] = stats_globales['defense_domicile'] / avg_a
    stats_globales['faibl_def_exterieur'] = stats_globales['defense_exterieur'] / avg_h
    
    return stats_globales, avg_h, avg_a

def predire_et_simuler(equipe_dom, equipe_ext, stats, avg_h, avg_a, n_simulations=10000):
    """Prédit le résultat d'un match"""
    
    force_att_dom = stats.loc[equipe_dom, 'force_att_domicile']
    faibl_def_ext = stats.loc[equipe_ext, 'faibl_def_exterieur']
    buts_projetes_dom = force_att_dom * faibl_def_ext * avg_h
    
    force_att_ext = stats.loc[equipe_ext, 'force_att_exterieur']
    faibl_def_dom = stats.loc[equipe_dom, 'faibl_def_domicile']
    buts_projetes_ext = force_att_ext * faibl_def_dom * avg_a
    
    # Simulation Monte Carlo
    buts_simules_dom = np.random.poisson(buts_projetes_dom, n_simulations)
    buts_simules_ext = np.random.poisson(buts_projetes_ext, n_simulations)
    
    victoires_dom = np.sum(buts_simules_dom > buts_simules_ext)
    nuls = np.sum(buts_simules_dom == buts_simules_ext)
    victoires_ext = np.sum(buts_simules_dom < buts_simules_ext)
    
    prob_1 = (victoires_dom / n_simulations) * 100
    prob_N = (nuls / n_simulations) * 100
    prob_2 = (victoires_ext / n_simulations) * 100
    
    return {
        'buts_dom': round(buts_projetes_dom, 2),
        'buts_ext': round(buts_projetes_ext, 2),
        'prob_1': round(prob_1, 1),
        'prob_N': round(prob_N, 1),
        'prob_2': round(prob_2, 1),
        'cote_1': round(100 / prob_1, 2) if prob_1 > 0 else float('inf'),
        'cote_N': round(100 / prob_N, 2) if prob_N > 0 else float('inf'),
        'cote_2': round(100 / prob_2, 2) if prob_2 > 0 else float('inf')
    }

# --- CHARGEMENT AU DÉMARRAGE ---
print("⏳ Chargement des données...")
df = charger_donnees()
print("⏳ Entraînement du modèle...")
stats_equipes, avg_home, avg_away = entrainer_modele(df, span=10)
equipes = sorted(stats_equipes.index.tolist())
print("✅ Modèle prêt!")

# --- TEMPLATE HTML ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ligue 1 Match Prediction</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
            background: #f5f7fa;
            color: #333;
            line-height: 1.6;
        }
        
        .header {
            background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
            color: white;
            padding: 40px 20px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 8px;
            font-weight: 700;
            letter-spacing: -0.5px;
        }
        
        .header p {
            font-size: 1.05em;
            opacity: 0.9;
            font-weight: 300;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px 20px;
        }
        
        .main-card {
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
            padding: 40px;
            margin-bottom: 30px;
        }
        
        .form-section {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }
        
        @media (max-width: 768px) {
            .form-section {
                grid-template-columns: 1fr;
            }
        }
        
        .form-group {
            display: flex;
            flex-direction: column;
        }
        
        label {
            font-size: 0.95em;
            font-weight: 600;
            color: #1e3a8a;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        select {
            padding: 12px 15px;
            border: 1.5px solid #e0e7ff;
            border-radius: 8px;
            font-size: 1em;
            cursor: pointer;
            transition: all 0.3s ease;
            background: white;
            color: #333;
            appearance: none;
            background-image: url("data:image/svg+xml;charset=UTF-8,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%231e3a8a' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3e%3cpolyline points='6 9 12 15 18 9'%3e%3c/polyline%3e%3c/svg%3e");
            background-repeat: no-repeat;
            background-position: right 10px center;
            background-size: 20px;
            padding-right: 40px;
        }
        
        select:hover {
            border-color: #1e3a8a;
            box-shadow: 0 0 0 3px rgba(30, 58, 138, 0.05);
        }
        
        select:focus {
            outline: none;
            border-color: #1e3a8a;
            box-shadow: 0 0 0 4px rgba(30, 58, 138, 0.1);
        }
        
        .button-section {
            text-align: center;
            margin-bottom: 0;
            display: flex;
            justify-content: center;
        }
        
        button {
            background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
            color: white;
            padding: 14px 45px;
            border: none;
            border-radius: 8px;
            font-size: 1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            min-width: 200px;
            letter-spacing: 0.5px;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(30, 58, 138, 0.25);
        }
        
        button:active {
            transform: translateY(0);
        }
        
        .results {
            display: none;
            animation: slideIn 0.5s ease;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .results.show {
            display: block;
        }
        
        .section-title {
            font-size: 1.2em;
            font-weight: 700;
            color: #1e3a8a;
            margin-bottom: 20px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .goals-section {
            background: linear-gradient(135deg, #f0f4ff 0%, #e8ecff 100%);
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 40px;
            display: grid;
            grid-template-columns: 1fr auto 1fr;
            gap: 30px;
            align-items: center;
        }
        
        @media (max-width: 768px) {
            .goals-section {
                grid-template-columns: 1fr;
            }
        }
        
        .goal-card {
            text-align: center;
        }
        
        .goal-value {
            font-size: 3.5em;
            font-weight: 800;
            color: #1e3a8a;
            line-height: 1;
        }
        
        .goal-label {
            color: #64748b;
            font-size: 0.95em;
            margin-top: 8px;
            font-weight: 500;
        }
        
        .vs-divider {
            text-align: center;
            color: #cbd5e1;
            font-size: 1.5em;
            font-weight: 300;
        }
        
        .probabilities {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-bottom: 40px;
        }
        
        @media (max-width: 768px) {
            .probabilities {
                grid-template-columns: 1fr;
            }
        }
        
        .prob-card {
            background: white;
            border: 1.5px solid #e0e7ff;
            padding: 25px;
            border-radius: 10px;
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .prob-card:hover {
            border-color: #1e3a8a;
            box-shadow: 0 4px 12px rgba(30, 58, 138, 0.12);
        }
        
        .prob-label {
            font-size: 0.9em;
            color: #64748b;
            font-weight: 500;
            margin-bottom: 8px;
        }
        
        .prob-value {
            font-size: 2.8em;
            font-weight: 800;
            color: #1e3a8a;
            line-height: 1;
            margin: 12px 0;
        }
        
        .cote {
            background: #f0f4ff;
            color: #1e3a8a;
            padding: 10px 12px;
            border-radius: 6px;
            font-weight: 700;
            font-size: 0.9em;
            display: inline-block;
            margin-top: 12px;
            border: 1px solid #e0e7ff;
        }
        
        .results-table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }
        
        .results-table th {
            background: #f0f4ff;
            color: #1e3a8a;
            padding: 16px;
            text-align: left;
            font-weight: 700;
            font-size: 0.95em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-bottom: 2px solid #e0e7ff;
        }
        
        .results-table td {
            padding: 16px;
            border-bottom: 1px solid #e0e7ff;
            color: #475569;
            font-weight: 500;
        }
        
        .results-table tr:hover {
            background: #f8fafc;
        }
        
        .results-table td:last-child {
            font-weight: 700;
            color: #1e3a8a;
            font-size: 1.05em;
        }
        
        .error {
            color: #991b1b;
            background: #fef2f2;
            padding: 15px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: none;
            border: 1px solid #fecaca;
            font-weight: 500;
        }
        
        .error.show {
            display: block;
        }
        
        .footer {
            background: white;
            border-radius: 12px;
            padding: 30px;
            margin-top: 40px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
            color: #64748b;
            font-size: 0.95em;
            line-height: 1.8;
        }
        
        .footer h3 {
            color: #1e3a8a;
            font-weight: 700;
            margin-bottom: 12px;
            text-transform: uppercase;
            font-size: 0.9em;
            letter-spacing: 0.5px;
        }
        
        .footer ul {
            list-style: none;
        }
        
        .footer li {
            padding: 4px 0;
        }
        
        .loading {
            display: none;
            text-align: center;
            color: #1e3a8a;
            padding: 20px;
        }
        
        .loading.show {
            display: block;
        }
        
        .spinner {
            border: 3px solid #f0f4ff;
            border-top: 3px solid #1e3a8a;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Ligue 1 Match Prediction</h1>
        <p>Forecast backed by EWMA analysis and Monte Carlo simulation</p>
    </div>
    
    <div class="container">
        <div class="main-card">
            <form id="predictionForm">
                <div class="form-section">
                    <div class="form-group">
                        <label for="home_team">Home Team</label>
                        <select id="home_team" name="home_team" required>
                            <option value="">Select a team</option>
                            {% for team in equipes %}
                            <option value="{{ team }}">{{ team }}</option>
                            {% endfor %}
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="away_team">Away Team</label>
                        <select id="away_team" name="away_team" required>
                            <option value="">Select a team</option>
                            {% for team in equipes %}
                            <option value="{{ team }}">{{ team }}</option>
                            {% endfor %}
                        </select>
                    </div>
                </div>
                
                <div class="button-section">
                    <button type="submit">Predict Match</button>
                </div>
            </form>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Analyzing match...</p>
            </div>
            
            <div class="error" id="errorMessage"></div>
        </div>
        
        <div class="results" id="results">
            <div class="main-card">
                <h2 class="section-title">Expected Goals</h2>
                <div class="goals-section" id="goalsSection"></div>
            </div>
            
            <div class="main-card">
                <h2 class="section-title">Outcome Probabilities</h2>
                <div class="probabilities" id="probabilities"></div>
            </div>
            
            <div class="main-card">
                <h2 class="section-title">Fair Odds</h2>
                <table class="results-table" id="coteTable">
                    <thead>
                        <tr>
                            <th>Outcome</th>
                            <th>Probability</th>
                            <th>Fair Odds</th>
                        </tr>
                    </thead>
                    <tbody id="coteBody"></tbody>
                </table>
            </div>
        </div>
        
        <div class="footer">
            <h3>Methodology</h3>
            <ul>
                <li>Model based on goals scored and conceded</li>
                <li>EWMA (span=10) to emphasize recent form</li>
                <li>Separation of home/away statistics per team</li>
                <li>Monte Carlo simulation using Poisson distribution</li>
            </ul>
        </div>
    </div>
    
    <script>
        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const homeTeam = document.getElementById('home_team').value;
            const awayTeam = document.getElementById('away_team').value;
            const errorDiv = document.getElementById('errorMessage');
            const resultsDiv = document.getElementById('results');
            const loadingDiv = document.getElementById('loading');
            
            errorDiv.classList.remove('show');
            resultsDiv.classList.remove('show');
            loadingDiv.classList.add('show');
            
            if (homeTeam === awayTeam) {
                errorDiv.textContent = 'Please select two different teams';
                errorDiv.classList.add('show');
                loadingDiv.classList.remove('show');
                return;
            }
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        home_team: homeTeam,
                        away_team: awayTeam
                    })
                });
                
                const data = await response.json();
                loadingDiv.classList.remove('show');
                
                if (data.error) {
                    errorDiv.textContent = data.error;
                    errorDiv.classList.add('show');
                    return;
                }
                
                const goalsHTML = `
                    <div class="goal-card">
                        <div class="goal-value">${data.buts_dom}</div>
                        <div class="goal-label">${homeTeam}</div>
                    </div>
                    <div class="vs-divider">VS</div>
                    <div class="goal-card">
                        <div class="goal-value">${data.buts_ext}</div>
                        <div class="goal-label">${awayTeam}</div>
                    </div>
                `;
                document.getElementById('goalsSection').innerHTML = goalsHTML;
                
                const probHTML = `
                    <div class="prob-card">
                        <div class="prob-label">${homeTeam} Win</div>
                        <div class="prob-value">${data.prob_1}%</div>
                        <div class="cote">${data.cote_1}</div>
                    </div>
                    <div class="prob-card">
                        <div class="prob-label">Draw</div>
                        <div class="prob-value">${data.prob_N}%</div>
                        <div class="cote">${data.cote_N}</div>
                    </div>
                    <div class="prob-card">
                        <div class="prob-label">${awayTeam} Win</div>
                        <div class="prob-value">${data.prob_2}%</div>
                        <div class="cote">${data.cote_2}</div>
                    </div>
                `;
                document.getElementById('probabilities').innerHTML = probHTML;
                
                const tableHTML = `
                    <tr>
                        <td>${homeTeam} Win</td>
                        <td>${data.prob_1}%</td>
                        <td>${data.cote_1}</td>
                    </tr>
                    <tr>
                        <td>Draw</td>
                        <td>${data.prob_N}%</td>
                        <td>${data.cote_N}</td>
                    </tr>
                    <tr>
                        <td>${awayTeam} Win</td>
                        <td>${data.prob_2}%</td>
                        <td>${data.cote_2}</td>
                    </tr>
                `;
                document.getElementById('coteBody').innerHTML = tableHTML;
                
                resultsDiv.classList.add('show');
            } catch (error) {
                errorDiv.textContent = 'Prediction error: ' + error.message;
                errorDiv.classList.add('show');
                loadingDiv.classList.remove('show');
            }
        });
    </script>
</body>
</html>
"""

# --- ROUTES ---
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, equipes=equipes)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        home_team = data.get('home_team')
        away_team = data.get('away_team')
        
        if not home_team or not away_team:
            return jsonify({'error': 'Équipes manquantes'}), 400
        
        if home_team == away_team:
            return jsonify({'error': 'Les deux équipes doivent être différentes'}), 400
        
        if home_team not in stats_equipes.index:
            return jsonify({'error': f"L'équipe '{home_team}' n'existe pas"}), 400
        
        if away_team not in stats_equipes.index:
            return jsonify({'error': f"L'équipe '{away_team}' n'existe pas"}), 400
        
        resultats = predire_et_simuler(home_team, away_team, stats_equipes, avg_home, avg_away)
        return jsonify(resultats)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import os
    
    # Déterminer le port et l'host
    port = int(os.getenv('PORT', 7860))  # HF Spaces utilise le port 7860
    host = '0.0.0.0'  # Écouter sur toutes les interfaces
    
    print(f"\n🚀 Serveur Flask démarré!")
    print(f"📍 Accédez à http://0.0.0.0:{port}")
    app.run(host=host, port=port, debug=False)
