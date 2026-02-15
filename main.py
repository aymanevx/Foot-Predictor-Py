import pandas as pd
import numpy as np


df = pd.read_csv('https://www.football-data.co.uk/mmz4281/2526/F1.csv')
df = df[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']]
df.columns = ['home_team', 'away_team', 'home_goals', 'away_goals']

# --- ASTUCE PRO : On "cap" les scores pour réduire l'impact des cartons ---
# Un 4-0 ou un 5-0 ne veut pas dire que l'équipe est 5x plus forte, 
# souvent l'adversaire a juste lâché l'affaire à 2-0.
df['home_goals_adj'] = df['home_goals'].clip(upper=3.5) 
df['away_goals_adj'] = df['away_goals'].clip(upper=3.5)

# Ajouter un index pour garder l'ordre chronologique
df['match_order'] = range(len(df))

def entrainer_modele_buts(df, span=10):
    """
    Entraîne le modèle avec EWMA (Exponentially Weighted Moving Average)
    span=10 : donne un poids plus important aux matchs récents
    """
    # Moyennes globales (toutes les données)
    avg_h = df['home_goals_adj'].mean()
    avg_a = df['away_goals_adj'].mean()
    
    stats_globales = pd.DataFrame()
    
    # Traiter DOMICILE : En groupant par home_team
    for team in df['home_team'].unique():
        mask = df['home_team'] == team
        home_data = df[mask].copy().sort_values('match_order')
        
        if len(home_data) > 0:
            # EWMA pour l'attaque et la défense à domicile
            att_ewma = home_data['home_goals_adj'].ewm(span=span).mean().iloc[-1]
            def_ewma = home_data['away_goals_adj'].ewm(span=span).mean().iloc[-1]
            
            stats_globales.loc[team, 'attaque_domicile'] = att_ewma
            stats_globales.loc[team, 'defense_domicile'] = def_ewma
    
    # Traiter EXTÉRIEUR : En groupant par away_team
    for team in df['away_team'].unique():
        mask = df['away_team'] == team
        away_data = df[mask].copy().sort_values('match_order')
        
        if len(away_data) > 0:
            # EWMA pour l'attaque et la défense à l'extérieur
            att_ewma = away_data['away_goals_adj'].ewm(span=span).mean().iloc[-1]
            def_ewma = away_data['home_goals_adj'].ewm(span=span).mean().iloc[-1]
            
            stats_globales.loc[team, 'attaque_exterieur'] = att_ewma
            stats_globales.loc[team, 'defense_exterieur'] = def_ewma
    
    # Remplir les valeurs manquantes avec les moyennes globales (si équipe sans match)
    stats_globales = stats_globales.fillna(avg_h)
    
    # Calcul des Forces (séparation domicile/extérieur)
    stats_globales['force_att_domicile'] = stats_globales['attaque_domicile'] / avg_h
    stats_globales['force_att_exterieur'] = stats_globales['attaque_exterieur'] / avg_a
    stats_globales['faibl_def_domicile'] = stats_globales['defense_domicile'] / avg_a
    stats_globales['faibl_def_exterieur'] = stats_globales['defense_exterieur'] / avg_h
    
    return stats_globales, avg_h, avg_a

# Le reste (Simulation) est IDENTIQUE au code précédent...
# La fonction de simulation attend juste des chiffres, elle se fiche de savoir si c'est des buts ou des xG.

# On lance l'entraînement avec EWMA (span=10)
stats_equipes, avg_home, avg_away = entrainer_modele_buts(df, span=10)

# Avantage domicile (coefficient empirique ~15% d'augmentation des buts)
home_advantage = 1.15

# ---------------------------------------------------------
# 3. LA PRÉDICTION ET LA SIMULATION
# ---------------------------------------------------------

def predire_et_simuler(equipe_dom, equipe_ext, stats, avg_h, avg_a, n_simulations=10000):
    
    # --- ÉTAPE A : Calcul des Buts Projetés (Attendus) ---
    # Utilise les forces séparées domicile/extérieur basées sur l'EWMA
    
    # Formule : (Force Attaque Domicile * Faiblesse Défense Extérieur) * Moyenne Domicile * Avantage Domicile
    force_att_dom = stats.loc[equipe_dom, 'force_att_domicile']
    faibl_def_ext = stats.loc[equipe_ext, 'faibl_def_exterieur']
    buts_projetes_dom = force_att_dom * faibl_def_ext * avg_h
    
    # Formule : (Force Attaque Extérieur * Faiblesse Défense Domicile) * Moyenne Extérieur
    force_att_ext = stats.loc[equipe_ext, 'force_att_exterieur']
    faibl_def_dom = stats.loc[equipe_dom, 'faibl_def_domicile']
    buts_projetes_ext = force_att_ext * faibl_def_dom * avg_a
    
    print(f"--- PRÉDICTION BUTS ATTENDUS ---")
    print(f"{equipe_dom} (buts attendus): {buts_projetes_dom:.2f}")
    print(f"{equipe_ext} (buts attendus): {buts_projetes_ext:.2f}")
    
    # --- ÉTAPE B : La Simulation Monte Carlo (Loi de Poisson) ---
    
    # On simule 10 000 matchs virtuels avec ces moyennes
    buts_simules_dom = np.random.poisson(buts_projetes_dom, n_simulations)
    buts_simules_ext = np.random.poisson(buts_projetes_ext, n_simulations)
    
    # On compte les résultats
    victoires_dom = np.sum(buts_simules_dom > buts_simules_ext)
    nuls = np.sum(buts_simules_dom == buts_simules_ext)
    victoires_ext = np.sum(buts_simules_dom < buts_simules_ext)
    
    # Conversion en pourcentages
    prob_1 = (victoires_dom / n_simulations) * 100
    prob_N = (nuls / n_simulations) * 100
    prob_2 = (victoires_ext / n_simulations) * 100
    
    return prob_1, prob_N, prob_2

# ---------------------------------------------------------
# 4. TEST DU MODÈLE
# ---------------------------------------------------------

try:
    # Prédiction : Lille vs Brest
    p1, pN, p2 = predire_et_simuler('Lille', 'Brest', stats_equipes, avg_home, avg_away)

    print(f"\n--- RÉSULTATS SIMULATION (10 000 matchs) ---")
    print(f"Victoire Lille: {p1:.1f}%")
    print(f"Match Nul :        {pN:.1f}%")
    print(f"Victoire Brest: {p2:.1f}%")
    
    # Calcul des cotes justes (Fair Odds)
    print(f"\n--- COTES DU MODÈLE ---")
    print(f"Cote 1 (Lille): {100/p1:.2f}")
    print(f"Cote N (Nul)     : {100/pN:.2f}")
    print(f"Cote 2 (Brest): {100/p2:.2f}")


except KeyError as e:
    print(f"Erreur : Une équipe n'est pas dans la base de données ({e})")
    print("\n--- ÉQUIPES DISPONIBLES ---")
    print(stats_equipes.index.tolist())