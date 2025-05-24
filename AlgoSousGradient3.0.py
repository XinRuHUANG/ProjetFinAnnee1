# -*- coding: utf-8 -*-
"""
Created on Fri May 23 22:45:22 2025

@author: Cytech
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools
import networkx as nx

def Generateur_Graphes(n=6, width=100, height=100, seed=None, show=True):
    """
    Génère un graphe PVC complet avec sommets aléatoires et distances euclidiennes.
    Retourne la matrice des distances et les coordonnées des sommets.
    """
    if seed is not None:
        np.random.seed(seed)

    coords = np.random.rand(n, 2) * [width, height]
    MatriceDistance = np.linalg.norm(coords[:, np.newaxis, :] - coords[np.newaxis, :, :], axis=-1)

    if show:
        plt.figure(figsize=(6, 6))
        for i, j in itertools.combinations(range(n), 2):
            x = [coords[i][0], coords[j][0]]
            y = [coords[i][1], coords[j][1]]
            w = round(MatriceDistance[i][j], 1)
            plt.plot(x, y, 'gray', linewidth=0.8)
            plt.text((x[0]+x[1])/2, (y[0]+y[1])/2, str(w), fontsize=7, color='darkred', ha='center')
        for i, (x, y) in enumerate(coords):
            plt.scatter(x, y, c='blue')
            plt.text(x + 1, y + 1, str(i), fontsize=9)
        plt.title("Graphe PVC : sommets + arêtes pondérées")
        plt.grid(True)
        plt.axis('equal')
        plt.show()

    return MatriceDistance, coords

def resoudre_relaxation (D, lambdas):
    """
    Résout le problème primal relaxé en minimisant le lagrangien L(x, lambda)
    """
    n = len(D)
    meilleur_cout = float('inf')
    meilleure_solution = None
    meilleur_cycle = None

    for ordre_villes in itertools.permutations(range(n)):
        cycle = list(ordre_villes) + [ordre_villes[0]]
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

def sous_gradient(D, nb_iterations=10, alpha_initial=1.0):
    """
    Applique la méthode du sous-gradient sur le dual du PVC relaxé.
    """
    n = len(D)
    lambdas = np.zeros(n)
    valeurs_duales = []
    cycles_optimaux = []

    for k in range(nb_iterations):
        matrice_x, cout_dual, cycle = resoudre_relaxation(D, lambdas)
        gradient = np.array([1 - sum(matrice_x[i]) for i in range(n)])
        alpha_k = alpha_initial / (k + 1)
        lambdas = np.maximum(0, lambdas + alpha_k * gradient)
        valeurs_duales.append(cout_dual)
        cycles_optimaux.append(cycle)

    return lambdas, valeurs_duales, cycles_optimaux

def visualiser_cycle_optimal (D, coords, cycle_optimal):
    """
    Affiche le graphe complet avec les arêtes pondérées et surligne le cycle optimal en rouge.
    """
    n = len(D)
    G = nx.Graph()

    for i in range(n):
        for j in range(i + 1, n):
            G.add_edge(i, j, weight=round(D[i][j], 1))
    pos = {i: (coords[i][0], coords[i][1]) for i in range(n)}

    couleurs_aretes = []
    liste_aretes = G.edges()
    aretes_optimales = {(cycle_optimal[i], cycle_optimal[i+1]) for i in range(len(cycle_optimal)-1)}
    aretes_optimales |= {(v, u) for u, v in aretes_optimales}

    for u, v in liste_aretes:
        if (u, v) in aretes_optimales or (v, u) in aretes_optimales:
            couleurs_aretes.append('red')
        else:
            couleurs_aretes.append('gray')

    plt.figure(figsize=(6, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color=couleurs_aretes, node_size=500, font_size=10)
    poids_aretes = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=poids_aretes, font_size=8)
    plt.title("Graphe des villes avec le chemin optimal en rouge")
    plt.axis('equal')
    plt.show()

# Exécution
matriceDistance, coords = Generateur_Graphes(n=6, seed=42, show=False)
lambdas_finaux, valeurs_duales, cycles_final = sous_gradient(matriceDistance)
visualiser_cycle_optimal(matriceDistance, coords, cycles_final[-1])
print("Matrice des distances :\n", matriceDistance)
print("Multiplicateurs finaux λ :", lambdas_finaux)
print("Valeurs duales :", valeurs_duales)
print("Cycle optimal trouvé :", cycles_final[-1])
