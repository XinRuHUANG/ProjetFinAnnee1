import numpy as np

def twoOptEchange(tour, i, j):
    """Effectue l'échange 2-opt entre les positions i et j"""
    nouveauTour = tour.copy()
    nouveauTour[i:j+1] = tour[i:j+1][::-1]  # Inverse la section du tour
    return nouveauTour

def twoOpt(tour, MatriceDistance, verbose=False):
    """Algorithme 2-opt corrigé"""
    n = len(tour)
    improvement = True
    iteration = 0

    while improvement:
        improvement = False
        for i in range(n-1):
            for j in range(i+2, n):  # i+2 pour éviter les arêtes adjacentes
                # Calcul des coûts
                old = MatriceDistance[tour[i]][tour[i+1]] + MatriceDistance[tour[j]][tour[(j+1)%n]]
                new = MatriceDistance[tour[i]][tour[j]] + MatriceDistance[tour[i+1]][tour[(j+1)%n]]

                if new < old:
                    tour[i+1:j+1] = tour[j:i:-1]  # Inversion plus efficace
                    improvement = True
                    if verbose:
                        print(f"Itération {iteration}: Gain = {old-new}")
                        print(f"Nouveau tour: {tour} (Coût: {calculCoutTotal(tour, MatriceDistance)})")
                    iteration += 1
    return tour

def calculCoutTotal(tour, MatriceDistance):
    """Calcule le coût total d'un tour"""
    return sum(MatriceDistance[tour[i]][tour[(i+1)%len(tour)]] for i in range(len(tour)))

def visualiserTour(tour, MatriceDistance):
    """Visualisation simple du tour"""
    print("\nVisualisation du tour:")
    total = 0
    for i in range(len(tour)):
        sommetCourant = tour[i]
        sommetSuivant = tour[(i+1)%len(tour)]
        cost = MatriceDistance[sommetCourant][sommetSuivant]
        total += cost
        print(f"{sommetCourant} → {sommetSuivant} (coût: {cost})")
    print(f"Coût total: {total}\n")

def twoOptPVC(MatriceDistance, tourInitial=None, verbose=False):
    """Interface principale pour l'algorithme 2-opt"""
    n = len(MatriceDistance)
    tour = tourInitial if tourInitial is not None else list(range(n))

    if verbose:
        print("Tour initial:", tour)
        visualiserTour(tour, MatriceDistance)

    tourOptimise = twoOpt(tour, MatriceDistance, verbose)
    coutTotal = calculCoutTotal(tourOptimise, MatriceDistance)

    return tourOptimise, coutTotal

# Exemple d'utilisation
# Matrice de distance pour un graphe à 4 sommets
MatriceDistance = np.array([
    [0, 2, 9, 10],
    [2, 0, 6, 4],
    [9, 6, 0, 8],
    [10, 4, 8, 0]
])

# Exécution avec affichage détaillé
tourFinal, coutFinal = twoOptPVC(MatriceDistance, verbose=True)
tourFinal = tourFinal + [tourFinal[0]]

print("\nRésultat final :")
print(f"Tour optimisé (liste simple)      : {tourFinal}")
print(f"Coût total du tour optimisé      : {coutFinal}")
