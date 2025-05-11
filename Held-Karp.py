import numpy as np
from itertools import combinations

def held_karp_tsp(distance_matrix):
    n = len(distance_matrix)
    
    # Dictionnaire pour stocker les coûts : clé = (ensemble de sommets, dernier sommet)
    memo = {}
    
    # Initialisation : coût pour aller du départ (0) à chaque autre sommet
    for k in range(1, n):
        memo[(frozenset([k]), k)] = distance_matrix[0][k]
    
    # Itérations pour des ensembles de taille croissante
    for subset_size in range(2, n):
        for subset in combinations(range(1, n), subset_size):
            subset = frozenset(subset)
            for k in subset:
                # Minimum sur tous les m dans subset\{k}
                min_cost = float('inf')
                for m in subset:
                    if m == k:
                        continue
                    cost = memo[(subset - {k}, m)] + distance_matrix[m][k]
                    if cost < min_cost:
                        min_cost = cost
                memo[(subset, k)] = min_cost
    
    # Calcul du tour complet
    full_set = frozenset(range(1, n))
    min_tour_cost = float('inf')
    for k in range(1, n):
        cost = memo[(full_set, k)] + distance_matrix[k][0]
        if cost < min_tour_cost:
            min_tour_cost = cost
    
    return min_tour_cost

# Exemple d'utilisation
if __name__ == "__main__":
    # Matrice de distance pour un graphe à 4 sommets
    dist_matrix = np.array([
        [0, 2, 9, 10],
        [2, 0, 6, 4],
        [9, 6, 0, 8],
        [10, 4, 8, 0]
    ])
    
    print("Borne inférieure Held-Karp:", held_karp_tsp(dist_matrix))