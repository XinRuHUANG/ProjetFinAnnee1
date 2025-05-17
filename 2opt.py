import numpy as np

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

# Exemple d'utilisation
# Matrice de distance pour un graphe à 4 sommets
matriceDistance = np.array([
    [0, 2, 9, 10],  # Ville 0
    [2, 0, 6, 4],   # Ville 1
    [9, 6, 0, 8],   # Ville 2
    [10, 4, 8, 0]   # Ville 3
])

# Exécution avec affichage détaillé
tourFinal, coutFinal = resoudrePVC2Opt(matriceDistance, verbose=True)

# Ajout du sommet initial pour affichage final
tourFinalComplet = tourFinal + [tourFinal[0]]

print("\nRésultat final:")
print(f"Tour optimal: {tourFinalComplet}")
print(f"Coût total: {coutFinal}")

# Affichage détaillé du tour final
visualiserTour(tourFinal, matriceDistance)