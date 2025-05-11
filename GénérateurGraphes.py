# -*- coding: utf-8 -*-
"""
Created on Sun May 11 23:02:30 2025

@author: Cytech
"""

import numpy as np
import matplotlib.pyplot as plt

def generate_distance_matrix_random(n, min_dist=1, max_dist=100, seed=None):
    """
    Génère une matrice de distances aléatoires symétrique pour un graphe TSP.
    """
    if seed is not None:
        np.random.seed(seed)
    mat = np.random.randint(min_dist, max_dist+1, size=(n, n))
    mat = (mat + mat.T) // 2  # Rendre la matrice symétrique
    np.fill_diagonal(mat, 0)  # Distance nulle entre un sommet et lui-même
    return mat

def generate_distance_matrix_coordinates(n, width=100, height=100, seed=None, show_plot=False):
    """
    Génère une matrice de distances à partir de points dans le plan (euclidiens).
    """
    if seed is not None:
        np.random.seed(seed)
    coords = np.random.rand(n, 2) * [width, height]
    mat = np.linalg.norm(coords[:, np.newaxis, :] - coords[np.newaxis, :, :], axis=-1)

    if show_plot:
        plt.figure(figsize=(6, 6))
        for i, (x, y) in enumerate(coords):
            plt.scatter(x, y, c='blue')
            plt.text(x + 1, y + 1, str(i), fontsize=9)
        plt.title("Graphe TSP : coordonnées aléatoires")
        plt.grid(True)
        plt.show()

    return mat
