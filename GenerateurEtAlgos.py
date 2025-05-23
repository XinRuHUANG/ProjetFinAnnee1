import numpy as np
import matplotlib.pyplot as plt
import itertools

def Generateur_Graphes(n=6, width=100, height=100, seed=None, show=True):
    """
    Génère un graphe PVC complet avec :
    - des sommets positionnés aléatoirement
    - des arêtes entre chaque paire de sommets avec poids (distance euclidienne)
    - la matrice d'adjacence renvoyée (pondérée, symétrique)
    """
    if seed is not None:
        np.random.seed(seed)

    # Coordonnées aléatoires des n sommets
    coords = np.random.rand(n, 2) * [width, height]

    # Matrice des distances euclidiennes entre tous les sommets
    MatriceDistance = np.linalg.norm(coords[:, np.newaxis, :] - coords[np.newaxis, :, :], axis=-1)

    if show:
        plt.figure(figsize=(6, 6))

        # Tracer toutes les arêtes (graphe complet)
        for i, j in itertools.combinations(range(n), 2):
            x = [coords[i][0], coords[j][0]]
            y = [coords[i][1], coords[j][1]]
            w = round(MatriceDistance[i][j], 1)
            plt.plot(x, y, 'gray', linewidth=0.8)
            plt.text((x[0]+x[1])/2, (y[0]+y[1])/2, str(w), fontsize=7, color='darkred', ha='center')

        # Tracer les sommets
        for i, (x, y) in enumerate(coords):
            plt.scatter(x, y, c='blue')
            plt.text(x + 1, y + 1, str(i), fontsize=9)

        plt.title("Graphe PVC : sommets + arêtes pondérées")
        plt.grid(True)
        plt.axis('equal')
        plt.show()

    # ✅ Retourne les données utiles pour la suite
    return MatriceDistance

def echange2Opt(tour, i, j):
    """Effectue l'échange 2-opt entre les positions i et j"""
    nouveauTour = tour.copy()
    nouveauTour[i:j+1] = tour[i:j+1][::-1]  # Inverse la section du tour
    return nouveauTour

def algorithme2Opt(tour, matriceDistance, verbose=False):
    """Algorithme 2-opt corrigé"""
    nombreVilles = len(tour)
    amelioration = True
    iteration = 0

    while amelioration:
        amelioration = False
        for i in range(nombreVilles-1):
            for j in range(i+2, nombreVilles):  # i+2 pour éviter les arêtes adjacentes
                # Calcul des coûts
                ancienCout = matriceDistance[tour[i]][tour[i+1]] + matriceDistance[tour[j]][tour[(j+1)%nombreVilles]]
                nouveauCout = matriceDistance[tour[i]][tour[j]] + matriceDistance[tour[i+1]][tour[(j+1)%nombreVilles]]

                if nouveauCout < ancienCout:
                    tour[i+1:j+1] = tour[j:i:-1]  # Inversion plus efficace
                    amelioration = True
                    if verbose:
                        print(f"Itération {iteration}: Gain = {ancienCout-nouveauCout}")
                        print(f"Nouveau tour: {tour} (Coût: {calculerCoutTotal(tour, matriceDistance)})")
                    iteration += 1
    return tour

def calculerCoutTotal(tour, matriceDistance):
    """Calcule le coût total d'un tour"""
    return sum(matriceDistance[tour[i]][tour[(i+1)%len(tour)]] for i in range(len(tour)))

def visualiserTour(tour, matriceDistance):
    """Visualisation simple du tour"""
    # Ajout du sommet initial pour former un cycle complet
    tourComplet = tour + [tour[0]]
    
    print("\nVisualisation du tour complet:")
    coutTotal = 0
    for i in range(len(tourComplet)-1):
        villeCourante = tourComplet[i]
        villeSuivante = tourComplet[i+1]
        cout = matriceDistance[villeCourante][villeSuivante]
        coutTotal += cout
        print(f"{villeCourante} → {villeSuivante} (coût: {cout})")
    print(f"Coût total du cycle: {coutTotal}\n")

def resoudrePVC2Opt(matriceDistance, tourInitial=None, verbose=False):
    """Interface principale pour l'algorithme 2-opt"""
    nombreVilles = len(matriceDistance)
    tour = tourInitial if tourInitial is not None else list(range(nombreVilles))

    if verbose:
        print("Tour initial:", tour)
        visualiserTour(tour, matriceDistance)

    tourOptimise = algorithme2Opt(tour, matriceDistance, verbose)
    coutTotal = calculerCoutTotal(tourOptimise, matriceDistance)

    # Retourne le tour optimisé et son coût
    return tourOptimise, coutTotal

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
    chemin.reverse()  # 0 ↔ … ↔ 0

    return optimum, chemin


matriceDistance = Generateur_Graphes(10,100,100,None,True)
print(matriceDistance)

# Exécution avec affichage détaillé
tourFinal, coutFinal = resoudrePVC2Opt(matriceDistance, verbose=True)

# Ajout du sommet initial pour affichage final
tourFinalComplet = tourFinal + [tourFinal[0]]

print("\nRésultat final:")
print(f"Tour optimal: {tourFinalComplet}")
print(f"Coût total: {coutFinal}")

# Affichage détaillé du tour final
visualiserTour(tourFinal, matriceDistance)

print("\Pour HeldKarp:")
coutOptimal, cheminOptimal = held_karp(matriceDistance)
print(f"Coût optimal: {coutOptimal}")
print(f"Chemin optimal: {' ↔ '.join(map(str, cheminOptimal))}")
# Exemple d'utilisation
