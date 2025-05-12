import numpy as np
from itertools import combinations

def HeldKarp(MatriceDistance):
    n = len(MatriceDistance)
    
    # Dictionnaires pour stocker les coûts et les chemins
    cout = {}
    chemin = {}
    
    # Initialisation
    for k in range(1, n):
        cout[(frozenset([k]), k)] = MatriceDistance[0][k]
        chemin[(frozenset([k]), k)] = [0, k]
    
    # Programmation dynamique
    for tailleEnsemble in range(2, n):
        for ensemble in combinations(range(1, n), tailleEnsemble):
            ensemble = frozenset(ensemble)
            for k in ensemble:
                minCout = float('inf')
                meilleurSommet = -1
                for m in ensemble:
                    if m == k:
                        continue
                    coutCourant = cout[(ensemble - {k}, m)] + MatriceDistance[m][k]
                    if coutCourant < minCout:
                        minCout = coutCourant
                        meilleurSommet = m
                cout[(ensemble, k)] = minCout
                chemin[(ensemble, k)] = chemin[(ensemble - {k}, meilleurSommet)] + [k]
    
    # Recherche de la solution optimale
    full_set = frozenset(range(1, n))
    minCout = float('inf')
    meilleurChemin = []
    
    for k in range(1, n):
        coutTotal = cout[(full_set, k)] + MatriceDistance[k][0]
        if coutTotal < minCout:
            minCout = coutTotal
            meilleurChemin = chemin[(full_set, k)] + [0]
    
    return minCout, meilleurChemin

# Exemple d'utilisation
MatriceDistance = np.array([
    [0, 2, 9, 10],
    [2, 0, 6, 4],
    [9, 6, 0, 8],
    [10, 4, 8, 0]
])

coutOptimal, cheminOptimal = HeldKarp(MatriceDistance)
print(f"Coût optimal: {coutOptimal}")
print(f"Chemin optimal: {' → '.join(map(str, cheminOptimal))}")