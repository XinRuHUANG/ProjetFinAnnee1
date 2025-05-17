# -*- coding: utf-8 -*-
"""
Created on Thu May 15 09:53:28 2025

@author: Cytech
"""

# Code complet de la méthode du sous-gradient appliquée au TSP relaxé (3 sommets)

import numpy as np
import itertools

def generate_MatriceDistance(n=3, seed=0):
    """
    Génère une matrice de distances symétrique avec zéros sur la diagonale.
    """
    np.random.seed(seed)
    mat = np.random.randint(1, 10, size=(n, n))
    np.fill_diagonal(mat, 0)
    return mat

def solve_primal_lagrangian(D, lambd):
    """
    Résout le problème primal relaxé : trouve le cycle (permutation) qui minimise
    le lagrangien L(x, lambda) à partir de la matrice D et des multiplicateurs lambda.
    """
    n = len(D)
    best_cost = float('inf')
    best_x = None
    best_cycle = None

    for perm in itertools.permutations(range(n)):
        cycle = list(perm) + [perm[0]]  # former un cycle fermé
        x = np.zeros((n, n))
        cost = 0
        for i in range(n):
            u, v = cycle[i], cycle[i + 1]
            x[u][v] = 1
            cost += D[u][v]
        lag_term = sum(lambd[i] * (1 - sum(x[i])) for i in range(n))
        total_cost = cost + lag_term

        if total_cost < best_cost:
            best_cost = total_cost
            best_x = x.copy()
            best_cycle = cycle

    return best_x, best_cost, best_cycle

def subgradient_method_verbose(D, max_iter=10, alpha_0=1.0):
    """
    Implémente la méthode du sous-gradient sur le dual du TSP relaxé.
    Suit l'évolution des lambda et des cycles sur plusieurs itérations.
    """
    n = len(D)
    lambd = np.zeros(n)
    dual_values = []
    best_cycles = []

    for k in range(max_iter):
        # Étape 1 : résolution du primal pour le lagrangien actuel
        x, cost, cycle = solve_primal_lagrangian(D, lambd)

        # Étape 2 : calcul du sous-gradient (1 sortie attendue par sommet)
        g = np.array([1 - sum(x[i]) for i in range(n)])

        # Étape 3 : mise à jour du pas
        alpha_k = alpha_0 / (k + 1)

        # Étape 4 : mise à jour de lambda
        lambd = np.maximum(0, lambd + alpha_k * g)

        # Enregistrement pour suivi
        dual_values.append(cost)
        best_cycles.append(cycle)

    return lambd, dual_values, best_cycles

# Exemple d'utilisation
D = generate_MatriceDistance(n=3, seed=147)
lambd_final, dual_values, best_cycles = subgradient_method_verbose(D)

D, lambd_final, dual_values, best_cycles[-1]
print("Matrice de distances :\n", D)
print("Multiplicateurs finaux λ :", lambd_final)
print("Valeurs duales :", dual_values)
print("Cycle optimal trouvé :", best_cycles[-1])
