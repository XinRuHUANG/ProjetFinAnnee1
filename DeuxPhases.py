import numpy as np

def construire_tableau_phase1(A, b, contraintes):
    m, n = A.shape
    slack_vars = 0
    art_vars = 0
    
    for c in contraintes:
        if c == '<=':
            slack_vars += 1
        elif c == '>=':
            slack_vars += 1  # variable d'excès en négatif comptée comme slack ici
            art_vars += 1
        elif c == '=':
            art_vars += 1
        else:
            raise ValueError(f"Contrainte non supportée : {c}")
    
    total_vars = n + slack_vars + art_vars
    tableau = np.zeros((m+1, total_vars+1))
    
    # Construction de la matrice augmentée
    slack_pos = n
    art_pos = n + slack_vars
    
    for i in range(m):
        tableau[i, :n] = A[i]
        if contraintes[i] == '<=':
            tableau[i, slack_pos] = 1
            slack_pos += 1
        elif contraintes[i] == '>=':
            tableau[i, slack_pos] = -1  # variable d'excès négative
            slack_pos += 1
            tableau[i, art_pos] = 1
            art_pos += 1
        elif contraintes[i] == '=':
            tableau[i, art_pos] = 1
            art_pos += 1
    
    tableau[:m, -1] = b
    
    # Objectif phase 1 : minimiser la somme des variables artificielles
    tableau[-1, n+slack_vars:n+slack_vars+art_vars] = 1
    
    return tableau, n, slack_vars, art_vars

def choisir_pivot(tableau):
    # Trouver la colonne pivot (plus négative dans la dernière ligne sauf RHS)
    last_row = tableau[-1, :-1]
    if np.all(last_row >= -1e-10):
        return -1, -1  # optimum atteint
    col_pivot = np.argmin(last_row)
    
    # Règle du minimum ratio pour ligne pivot
    ratios = []
    for i in range(len(tableau)-1):
        if tableau[i, col_pivot] > 1e-10:
            ratios.append(tableau[i, -1] / tableau[i, col_pivot])
        else:
            ratios.append(np.inf)
    if all(r == np.inf for r in ratios):
        raise ValueError("Problème non borné")
    ligne_pivot = np.argmin(ratios)
    return ligne_pivot, col_pivot

def pivot(tableau, ligne, colonne):
    pivot_val = tableau[ligne, colonne]
    tableau[ligne, :] /= pivot_val
    for i in range(len(tableau)):
        if i != ligne:
            tableau[i, :] -= tableau[i, colonne] * tableau[ligne, :]

def phase_simplexe(tableau, base_vars):
    while True:
        ligne, col = choisir_pivot(tableau)
        if ligne == -1:
            break
        pivot(tableau, ligne, col)
        base_vars[ligne] = col
    return tableau, base_vars

def extraire_solution(tableau, base_vars, n):
    x = np.zeros(n)
    for i, var in enumerate(base_vars):
        if var < n:
            x[var] = tableau[i, -1]
    return x

def methode_deux_phases(A, b, c, contraintes, objectif='max'):
    tableau, n, slack_vars, art_vars = construire_tableau_phase1(A, b, contraintes)
    m = len(b)
    
    # Base initiale : variables slack et artificielles (indices)
    base_vars = []
    slack_start = n
    art_start = n + slack_vars
    
    s = slack_start
    a = art_start
    
    for ct in contraintes:
        if ct == '<=':
            base_vars.append(s)
            s += 1
        elif ct == '>=':
            base_vars.append(a)
            a += 1
        elif ct == '=':
            base_vars.append(a)
            a += 1
    
    # Phase 1 : minimiser somme variables artificielles
    tableau, base_vars = phase_simplexe(tableau, base_vars)
    
    if abs(tableau[-1, -1]) > 1e-8:
        raise ValueError("Pas de solution réalisable (phase 1 non nulle)")
    
    # Supprimer colonnes des variables artificielles du tableau
    if art_vars > 0:
        art_indices = list(range(n + slack_vars, n + slack_vars + art_vars))
        tableau = np.delete(tableau, art_indices, axis=1)
        # Mise à jour total variables
        total_vars = tableau.shape[1] - 1
        # Mettre à jour base_vars : enlever celles artificielles (si existantes)
        base_vars = [v if v < art_indices[0] else -1 for v in base_vars]
    
    # Construire la nouvelle fonction objectif pour phase 2
    obj = np.zeros(tableau.shape[1]-1)
    for i in range(n):
        obj[i] = -c[i] if objectif == 'max' else c[i]
    
    tableau[-1, :-1] = obj
    
    # Ajuster la ligne objectif en fonction des variables de base
    for i in range(m):
        var = base_vars[i]
        if var != -1:
            tableau[-1, :] += tableau[-1, var] * tableau[i, :]
    
    # Phase 2 : optimiser le vrai objectif
    tableau, base_vars = phase_simplexe(tableau, base_vars)
    
    solution = extraire_solution(tableau, base_vars, n)
    valeur_obj = tableau[-1, -1]
    if objectif == 'max':
        valeur_obj = -valeur_obj
    
    return solution, valeur_obj

if __name__ == "__main__":
    A = np.array([
        [1, 1],
        [1, -1]
    ])
    b = np.array([4, 1])
    c = [3, 2]
    contraintes = ['<=', '<=']
    
    sol, val = methode_deux_phases(A, b, c, contraintes, objectif='max')
    print("Solution optimale:", sol)
    print("Valeur optimale:", val)
