import numpy as np
from itertools import combinations

### ALGOTHIME DE HELD-KARP

def HeldKarp(MatriceDistance):
    n = len(MatriceDistance)
    
    # Dictionnaire pour stocker les coûts : clé = (ensemble de sommets, dernier sommet)
    cout = {}
    
    # Initialisation : coût pour aller du départ (0) à chaque autre sommet
    for k in range(1, n):
        cout[(frozenset([k]), k)] = MatriceDistance[0][k]
    
    # Itérations pour des ensembles de taille croissante
    for tailleEnsemble in range(2, n):
        for ensemble in combinations(range(1, n), tailleEnsemble):
            ensemble = frozenset(ensemble)
            for k in ensemble:
                # Minimum sur tous les m dans ensemble\{k}
                minCout = float('inf')
                for m in ensemble:
                    if m == k:
                        continue
                    cout = cout[(ensemble - {k}, m)] + MatriceDistance[m][k]
                    if cout < minCout:
                        minCout = cout
                cout[(ensemble, k)] = minCout
    
    # Calcul du tour complet
    full_set = frozenset(range(1, n))
    minCoutTour = float('inf')
    for k in range(1, n):
        cout = cout[(full_set, k)] + MatriceDistance[k][0]
        if cout < minCoutTour:
            minCoutTour = cout
    
    return minCoutTour

### EXEMPLE D'UTILISATION

# Matrice de distance pour un graphe à 4 sommets
MatriceDistance = np.array([
    [0, 2, 9, 10],
    [2, 0, 6, 4],
    [9, 6, 0, 8],
    [10, 4, 8, 0]
])
    
print("Borne inférieure Held-Karp:", HeldKarp(MatriceDistance))