from flask import Flask, render_template_string, request, jsonify
import pandas as pd
import numpy as np

app = Flask(__name__)

CHAMPIONNATS = {
    'F1': 'Ligue 1 (France)',
    'E0': 'Premier League (Angleterre)',
    'SP1': 'LaLiga (Espagne)',
    'D1': 'Bundesliga (Allemagne)',
    'I1': 'Serie A (Italie)'
}
CHAMPIONNAT_DEFAUT = 'F1'
MODELES_CHAMPIONNAT = {}

# --- CHARGEMENT DES DONNÉES ---
def charger_donnees(championnat=CHAMPIONNAT_DEFAUT):
    if championnat not in CHAMPIONNATS:
        raise ValueError("Championnat invalide")

    df = pd.read_csv(f'https://www.football-data.co.uk/mmz4281/2526/{championnat}.csv')
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

def charger_modele_championnat(championnat=CHAMPIONNAT_DEFAUT, force_reload=False):
    if championnat not in CHAMPIONNATS:
        raise ValueError("Championnat invalide")

    if force_reload or championnat not in MODELES_CHAMPIONNAT:
        df = charger_donnees(championnat)
        stats_equipes, avg_home, avg_away = entrainer_modele(df, span=10)
        equipes = sorted(stats_equipes.index.tolist())
        MODELES_CHAMPIONNAT[championnat] = {
            'stats_equipes': stats_equipes,
            'avg_home': avg_home,
            'avg_away': avg_away,
            'equipes': equipes
        }

    return MODELES_CHAMPIONNAT[championnat]

# --- CHARGEMENT AU DÉMARRAGE ---
print(f"⏳ Chargement des données {CHAMPIONNATS[CHAMPIONNAT_DEFAUT]}...")
modele_defaut = charger_modele_championnat(CHAMPIONNAT_DEFAUT)
stats_equipes = modele_defaut['stats_equipes']
avg_home = modele_defaut['avg_home']
avg_away = modele_defaut['avg_away']
equipes = modele_defaut['equipes']
print("✅ Modèle prêt!")

# --- TEMPLATE HTML ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prédictions Football Top 5</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
            background: #f7f9fc;
            color: #0f172a;
            line-height: 1.6;
        }

        .header {
            background: #0f172a;
            color: white;
            padding: 42px 20px 36px;
            text-align: center;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        }

        .header h1 {
            font-size: 2rem;
            margin-bottom: 6px;
            font-weight: 750;
            letter-spacing: -0.02em;
        }

        .header p {
            font-size: 0.98rem;
            opacity: 0.86;
            font-weight: 400;
        }

        .container {
            max-width: 980px;
            margin: 0 auto;
            padding: 28px 18px 48px;
        }

        .main-card {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
            padding: 28px;
            margin-bottom: 18px;
        }

        .form-title {
            font-size: 1.05rem;
            color: #0f172a;
            margin-bottom: 16px;
            font-weight: 700;
        }

        .form-section {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-bottom: 18px;
        }

        @media (max-width: 768px) {
            .form-section {
                grid-template-columns: 1fr;
            }

            .main-card {
                padding: 20px;
            }

            .button-section {
                flex-direction: column;
            }

            button {
                width: 100%;
            }
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        label {
            font-size: 0.79rem;
            font-weight: 600;
            color: #334155;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        select {
            padding: 12px 14px;
            border: 1px solid #cbd5e1;
            border-radius: 10px;
            font-size: 0.98rem;
            cursor: pointer;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
            background: white;
            color: #0f172a;
            appearance: none;
            background-image: url("data:image/svg+xml;charset=UTF-8,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23334155' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3e%3cpolyline points='6 9 12 15 18 9'%3e%3c/polyline%3e%3c/svg%3e");
            background-repeat: no-repeat;
            background-position: right 12px center;
            background-size: 18px;
            padding-right: 40px;
        }

        select:hover {
            border-color: #64748b;
        }

        select:focus {
            outline: none;
            border-color: #0f172a;
            box-shadow: 0 0 0 3px rgba(15, 23, 42, 0.12);
        }

        .button-section {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }

        button {
            background: #0f172a;
            color: white;
            padding: 12px 18px;
            border: none;
            border-radius: 10px;
            font-size: 0.95rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
            min-width: 210px;
            letter-spacing: 0.02em;
        }

        button:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.22);
        }

        button:active {
            transform: translateY(0);
        }

        .secondary-button {
            background: #ffffff;
            color: #0f172a;
            border: 1px solid #cbd5e1;
        }

        .secondary-button:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.12);
        }

        .info {
            color: #0f172a;
            background: #f8fafc;
            padding: 12px 14px;
            border-radius: 8px;
            margin-top: 14px;
            display: none;
            border: 1px solid #cbd5e1;
            font-weight: 500;
            font-size: 0.92rem;
        }

        .info.show {
            display: block;
        }

        .results {
            display: none;
            animation: slideIn 0.5s ease;
        }

        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(10px);
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
            font-size: 0.82rem;
            font-weight: 700;
            color: #475569;
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .goals-section {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            padding: 22px;
            border-radius: 12px;
            display: grid;
            grid-template-columns: 1fr auto 1fr;
            gap: 16px;
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
            font-size: 2.8rem;
            font-weight: 800;
            color: #0f172a;
            line-height: 1;
        }

        .goal-label {
            color: #64748b;
            font-size: 0.92rem;
            margin-top: 6px;
            font-weight: 500;
        }

        .vs-divider {
            text-align: center;
            color: #94a3b8;
            font-size: 1rem;
            font-weight: 700;
            letter-spacing: 0.08em;
        }

        .probabilities {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
        }

        @media (max-width: 768px) {
            .probabilities {
                grid-template-columns: 1fr;
            }
        }

        .prob-card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            padding: 16px;
            border-radius: 12px;
            text-align: center;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }

        .prob-card:hover {
            border-color: #94a3b8;
            box-shadow: 0 6px 16px rgba(15, 23, 42, 0.08);
        }

        .prob-label {
            font-size: 0.86rem;
            color: #64748b;
            font-weight: 500;
            margin-bottom: 4px;
        }

        .prob-value {
            font-size: 2rem;
            font-weight: 800;
            color: #0f172a;
            line-height: 1;
            margin: 8px 0;
        }

        .cote {
            background: #ffffff;
            color: #0f172a;
            padding: 8px 10px;
            border-radius: 6px;
            font-weight: 700;
            font-size: 0.83rem;
            display: inline-block;
            margin-top: 8px;
            border: 1px solid #cbd5e1;
        }

        .results-table {
            width: 100%;
            border-collapse: collapse;
            border-radius: 10px;
            overflow: hidden;
        }

        .results-table th {
            background: #f8fafc;
            color: #334155;
            padding: 12px;
            text-align: left;
            font-weight: 700;
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            border-bottom: 1px solid #e2e8f0;
        }

        .results-table td {
            padding: 12px;
            border-bottom: 1px solid #e2e8f0;
            color: #475569;
            font-weight: 500;
            font-size: 0.92rem;
        }

        .results-table tr:hover {
            background: #f8fafc;
        }

        .results-table td:last-child {
            font-weight: 700;
            color: #0f172a;
            font-size: 0.95rem;
        }

        .error {
            color: #991b1b;
            background: #fef2f2;
            padding: 12px 14px;
            border-radius: 8px;
            margin-top: 14px;
            display: none;
            border: 1px solid #fecaca;
            font-weight: 500;
            font-size: 0.92rem;
        }

        .error.show {
            display: block;
        }

        .footer {
            background: white;
            border-radius: 14px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
            padding: 22px;
            margin-top: 10px;
            color: #64748b;
            font-size: 0.9rem;
            line-height: 1.7;
        }

        .footer h3 {
            color: #334155;
            font-weight: 700;
            margin-bottom: 10px;
            text-transform: uppercase;
            font-size: 0.76rem;
            letter-spacing: 0.08em;
        }

        .footer ul {
            list-style: none;
        }

        .footer li {
            padding: 2px 0;
        }

        .loading {
            display: none;
            text-align: center;
            color: #334155;
            padding: 16px;
        }

        .loading.show {
            display: block;
        }

        .spinner {
            border: 3px solid #e2e8f0;
            border-top: 3px solid #0f172a;
            border-radius: 50%;
            width: 26px;
            height: 26px;
            animation: spin 1s linear infinite;
            margin: 0 auto 8px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Prédictions Football Top 5</h1>
        <p>Interface simple, rapide et claire pour simuler vos matchs</p>
    </div>
    
    <div class="container">
        <div class="main-card">
            <h2 class="form-title">Paramètres du match</h2>
            <form id="predictionForm">
                <div class="form-group" style="margin-bottom: 18px;">
                    <label for="league">Championnat</label>
                    <select id="league" name="league" required>
                        {% for code, nom in championnats.items() %}
                        <option value="{{ code }}" {% if code == championnat_defaut %}selected{% endif %}>{{ nom }}</option>
                        {% endfor %}
                    </select>
                </div>

                <div class="form-section">
                    <div class="form-group">
                        <label for="home_team">Équipe à domicile</label>
                        <select id="home_team" name="home_team" required>
                            <option value="">Sélectionner une équipe</option>
                            {% for team in equipes %}
                            <option value="{{ team }}">{{ team }}</option>
                            {% endfor %}
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="away_team">Équipe à l'extérieur</label>
                        <select id="away_team" name="away_team" required>
                            <option value="">Sélectionner une équipe</option>
                            {% for team in equipes %}
                            <option value="{{ team }}">{{ team }}</option>
                            {% endfor %}
                        </select>
                    </div>
                </div>
                
                <div class="button-section">
                    <button type="submit">Prédire le match</button>
                    <button type="button" id="refreshBtn" class="secondary-button">Actualiser les données</button>
                </div>
            </form>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Analyse du match...</p>
            </div>
            
            <div class="info" id="infoMessage"></div>
            <div class="error" id="errorMessage"></div>
        </div>
        
        <div class="results" id="results">
            <div class="main-card">
                <h2 class="section-title">Buts attendus</h2>
                <div class="goals-section" id="goalsSection"></div>
            </div>
            
            <div class="main-card">
                <h2 class="section-title">Probabilités de résultat</h2>
                <div class="probabilities" id="probabilities"></div>
            </div>
            
            <div class="main-card">
                <h2 class="section-title">Cotes justes</h2>
                <table class="results-table" id="coteTable">
                    <thead>
                        <tr>
                            <th>Résultat</th>
                            <th>Probabilité</th>
                            <th>Cote juste</th>
                        </tr>
                    </thead>
                    <tbody id="coteBody"></tbody>
                </table>
            </div>
        </div>
        
        <div class="footer">
            <h3>Méthodologie</h3>
            <ul>
                <li>Data en temps réel depuis football-data.co.uk</li>
                <li>EWMA (span=10) pour privilégier la forme récente</li>
                <li>Statistiques domicile/extérieur séparées</li>
                <li>Simulation Monte Carlo (distribution de Poisson)</li>
            </ul>
        </div>
    </div>
    
    <script>
        function remplirEquipes(equipes) {
            const homeSelect = document.getElementById('home_team');
            const awaySelect = document.getElementById('away_team');

            const defaultOptionHome = '<option value="">Sélectionner une équipe</option>';
            const defaultOptionAway = '<option value="">Sélectionner une équipe</option>';
            const options = equipes.map((team) => `<option value="${team}">${team}</option>`).join('');

            homeSelect.innerHTML = defaultOptionHome + options;
            awaySelect.innerHTML = defaultOptionAway + options;
        }

        async function chargerEquipes(championnat, forceReload = false) {
            const infoDiv = document.getElementById('infoMessage');
            const errorDiv = document.getElementById('errorMessage');
            const endpoint = forceReload ? '/refresh' : '/teams';

            infoDiv.classList.remove('show');
            errorDiv.classList.remove('show');

            const response = await fetch(`${endpoint}?league=${encodeURIComponent(championnat)}`);
            const data = await response.json();

            if (!response.ok || data.error) {
                throw new Error(data.error || 'Erreur lors du chargement des équipes');
            }

            remplirEquipes(data.equipes);

            if (forceReload) {
                infoDiv.textContent = `CSV actualisé : ${data.league_name}`;
                infoDiv.classList.add('show');
            }
        }

        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const league = document.getElementById('league').value;
            const homeTeam = document.getElementById('home_team').value;
            const awayTeam = document.getElementById('away_team').value;
            const infoDiv = document.getElementById('infoMessage');
            const errorDiv = document.getElementById('errorMessage');
            const resultsDiv = document.getElementById('results');
            const loadingDiv = document.getElementById('loading');
            
            infoDiv.classList.remove('show');
            errorDiv.classList.remove('show');
            resultsDiv.classList.remove('show');
            loadingDiv.classList.add('show');
            
            if (homeTeam === awayTeam) {
                errorDiv.textContent = 'Veuillez sélectionner deux équipes différentes';
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
                        league: league,
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
                        <div class="prob-label">Victoire ${homeTeam}</div>
                        <div class="prob-value">${data.prob_1}%</div>
                        <div class="cote">${data.cote_1}</div>
                    </div>
                    <div class="prob-card">
                        <div class="prob-label">Match nul</div>
                        <div class="prob-value">${data.prob_N}%</div>
                        <div class="cote">${data.cote_N}</div>
                    </div>
                    <div class="prob-card">
                        <div class="prob-label">Victoire ${awayTeam}</div>
                        <div class="prob-value">${data.prob_2}%</div>
                        <div class="cote">${data.cote_2}</div>
                    </div>
                `;
                document.getElementById('probabilities').innerHTML = probHTML;
                
                const tableHTML = `
                    <tr>
                        <td>Victoire ${homeTeam}</td>
                        <td>${data.prob_1}%</td>
                        <td>${data.cote_1}</td>
                    </tr>
                    <tr>
                        <td>Match nul</td>
                        <td>${data.prob_N}%</td>
                        <td>${data.cote_N}</td>
                    </tr>
                    <tr>
                        <td>Victoire ${awayTeam}</td>
                        <td>${data.prob_2}%</td>
                        <td>${data.cote_2}</td>
                    </tr>
                `;
                document.getElementById('coteBody').innerHTML = tableHTML;
                
                resultsDiv.classList.add('show');
            } catch (error) {
                errorDiv.textContent = 'Erreur de prédiction : ' + error.message;
                errorDiv.classList.add('show');
                loadingDiv.classList.remove('show');
            }
        });

        document.getElementById('league').addEventListener('change', async (e) => {
            const selectedLeague = e.target.value;
            const errorDiv = document.getElementById('errorMessage');
            const infoDiv = document.getElementById('infoMessage');
            const resultsDiv = document.getElementById('results');

            try {
                await chargerEquipes(selectedLeague, false);
                resultsDiv.classList.remove('show');
                infoDiv.classList.remove('show');
                errorDiv.classList.remove('show');
            } catch (error) {
                errorDiv.textContent = error.message;
                errorDiv.classList.add('show');
            }
        });

        document.getElementById('refreshBtn').addEventListener('click', async () => {
            const selectedLeague = document.getElementById('league').value;
            const refreshBtn = document.getElementById('refreshBtn');
            const errorDiv = document.getElementById('errorMessage');
            const resultsDiv = document.getElementById('results');

            refreshBtn.disabled = true;
            refreshBtn.textContent = 'Actualisation...';
            errorDiv.classList.remove('show');

            try {
                await chargerEquipes(selectedLeague, true);
                resultsDiv.classList.remove('show');
            } catch (error) {
                errorDiv.textContent = error.message;
                errorDiv.classList.add('show');
            } finally {
                refreshBtn.disabled = false;
                refreshBtn.textContent = 'Actualiser les données';
            }
        });
    </script>
</body>
</html>
"""

# --- ROUTES ---
@app.route('/')
def index():
    modele = charger_modele_championnat(CHAMPIONNAT_DEFAUT)
    return render_template_string(
        HTML_TEMPLATE,
        equipes=modele['equipes'],
        championnats=CHAMPIONNATS,
        championnat_defaut=CHAMPIONNAT_DEFAUT
    )

@app.route('/teams')
def teams():
    try:
        championnat = request.args.get('league', CHAMPIONNAT_DEFAUT)
        modele = charger_modele_championnat(championnat)
        return jsonify({
            'league': championnat,
            'league_name': CHAMPIONNATS[championnat],
            'equipes': modele['equipes']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/refresh')
def refresh():
    try:
        championnat = request.args.get('league', CHAMPIONNAT_DEFAUT)
        modele = charger_modele_championnat(championnat, force_reload=True)
        return jsonify({
            'league': championnat,
            'league_name': CHAMPIONNATS[championnat],
            'equipes': modele['equipes']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        championnat = data.get('league', CHAMPIONNAT_DEFAUT)
        home_team = data.get('home_team')
        away_team = data.get('away_team')

        if championnat not in CHAMPIONNATS:
            return jsonify({'error': 'Championnat invalide'}), 400

        modele = charger_modele_championnat(championnat)
        stats_equipes = modele['stats_equipes']
        avg_home = modele['avg_home']
        avg_away = modele['avg_away']
        
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
