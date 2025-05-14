import numpy as np
from itertools import combinations
from typing import Dict, FrozenSet, Tuple, List

def held_karp(D: np.ndarray) -> Tuple[float, List[int]]:
    """
    Renvoie (coût optimal, chemin optimal commençant et finissant au sommet 0).
    """
    n: int = D.shape[0]

    # c[(S, j)] : coût minimal pour partir de 0, visiter S⊆{1,…,n-1}, terminer en j∈S
    # p[(S, j)] : prédécesseur optimal de j
    c: Dict[Tuple[FrozenSet[int], int], float] = {}
    p: Dict[Tuple[FrozenSet[int], int], int] = {}

    # 1) Initialisation : S = {j}
    for j in range(1, n):
        S = frozenset([j])
        c[(S, j)] = D[0, j]       # c({j},j) = d_{0j}
        p[(S, j)] = 0             # prédécesseur immédiat

    # 2) Récurrence : 2 ≤ |S| ≤ n-1
    for size in range(2, n):
        for S in map(frozenset, combinations(range(1, n), size)):
            for j in S:
                S_minus_j = S - {j}
                pred, best = min(
                    ((i, c[(S_minus_j, i)] + D[i, j]) for i in S_minus_j),
                    key=lambda t: t[1]
                )
                c[(S, j)] = best
                p[(S, j)] = pred

    # 3) Retour au sommet 0
    full = frozenset(range(1, n))
    j_star, optimum = min(
        ((j, c[(full, j)] + D[j, 0]) for j in range(1, n)),
        key=lambda t: t[1]
    )

    # 4) Reconstruction du circuit
    chemin: List[int] = [0, j_star]
    S, j = full, j_star
    while S:
        i = p[(S, j)]
        chemin.append(i)
        S, j = S - {j}, i
    chemin.reverse()  # 0 → … → 0

    return optimum, chemin

# Exemple d'utilisation
MatriceDistance = np.array([
    [0, 2, 9, 10],
    [2, 0, 6, 4],
    [9, 6, 0, 8],
    [10, 4, 8, 0]
])

coutOptimal, cheminOptimal = held_karp(MatriceDistance)
print(f"Coût optimal: {coutOptimal}")
print(f"Chemin optimal: {' → '.join(map(str, cheminOptimal))}")
