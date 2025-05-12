# -*- coding: utf-8 -*-
"""
Created on Mon May 12 11:19:10 2025

@author: Cytech
"""

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
    dist_matrix = np.linalg.norm(coords[:, np.newaxis, :] - coords[np.newaxis, :, :], axis=-1)

    if show:
        plt.figure(figsize=(6, 6))

        # Tracer toutes les arêtes (graphe complet)
        for i, j in itertools.combinations(range(n), 2):
            x = [coords[i][0], coords[j][0]]
            y = [coords[i][1], coords[j][1]]
            w = round(dist_matrix[i][j], 1)
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
    return coords, dist_matrix

