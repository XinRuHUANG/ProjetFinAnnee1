import numpy as np
import itertools

def generer_matrice_distances(nb_villes=3, graine=0):
    """
    Génère une matrice de distances symétrique avec des zéros sur la diagonale.
    """
    np.random.seed(graine)
    matrice = np.random.randint(1, 10, size=(nb_villes, nb_villes))
    np.fill_diagonal(matrice, 0)
    return matrice

def resoudre_probleme_primal(D, lambdas):
    """
    Résout le problème primal relaxé : trouve le cycle qui minimise
    le lagrangien L(x, lambda) à partir de la matrice de distances D
    et des multiplicateurs de Lagrange.
    """
    n = len(D)
    meilleur_cout = float('inf')
    meilleure_solution = None
    meilleur_cycle = None

    for permutation in itertools.permutations(range(n)):
        cycle = list(permutation) + [permutation[0]]  # cycle fermé
        matrice_x = np.zeros((n, n))
        cout_cycle = 0

        for i in range(n):
            u, v = cycle[i], cycle[i + 1]
            matrice_x[u][v] = 1
            cout_cycle += D[u][v]

        terme_lagrangien = sum(lambdas[i] * (1 - sum(matrice_x[i])) for i in range(n))
        cout_total = cout_cycle + terme_lagrangien

        if cout_total < meilleur_cout:
            meilleur_cout = cout_total
            meilleure_solution = matrice_x.copy()
            meilleur_cycle = cycle

    return meilleure_solution, meilleur_cout, meilleur_cycle

def methode_du_sous_gradient(D, nb_iterations=10, alpha_initial=1.0):
    """
    Implémente la méthode du sous-gradient sur le problème dual relaxé du TSP.
    Suivi de l'évolution des multiplicateurs de Lagrange et des cycles.
    """
    n = len(D)
    lambdas = np.zeros(n)
    valeurs_duales = []
    cycles_optimaux = []

    for k in range(nb_iterations):
        # Étape 1 : résolution du problème primal
        matrice_x, cout_dual, cycle = resoudre_probleme_primal(D, lambdas)

        # Étape 2 : calcul du sous-gradient
        gradient = np.array([1 - sum(matrice_x[i]) for i in range(n)])

        # Étape 3 : mise à jour du pas
        alpha_k = alpha_initial / (k + 1)

        # Étape 4 : mise à jour des multiplicateurs
        lambdas = np.maximum(0, lambdas + alpha_k * gradient)

        # Enregistrement des résultats
        valeurs_duales.append(cout_dual)
        cycles_optimaux.append(cycle)

    return lambdas, valeurs_duales, cycles_optimaux

# Exemple d'utilisation
matrice_D = generer_matrice_distances(nb_villes=3, graine=147)
lambdas_finaux, valeurs_duales, cycles_final = methode_du_sous_gradient(matrice_D)

# Affichage des résultats
print("Matrice des distances :\n", matrice_D)
print("Multiplicateurs finaux λ :", lambdas_finaux)
print("Valeurs duales :", valeurs_duales)
print("Cycle optimal trouvé :", cycles_final[-1])
