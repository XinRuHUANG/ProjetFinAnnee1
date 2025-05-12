# -*- coding: utf-8 -*-
"""
Created on Sun May 11 23:02:30 2025

@author: Cytech
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools

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


def generate_weighted_graph_plot(n=6, width=100, height=100, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # Coordonnées aléatoires des sommets
    coords = np.random.rand(n, 2) * [width, height]

    # Matrice des distances euclidiennes
    dist_matrix = np.linalg.norm(coords[:, np.newaxis, :] - coords[np.newaxis, :, :], axis=-1)

    # Affichage du graphe
    plt.figure(figsize=(6, 6))
    
    # Afficher les arêtes avec les poids
    for i, j in itertools.combinations(range(n), 2):
        x = [coords[i][0], coords[j][0]]
        y = [coords[i][1], coords[j][1]]
        w = round(dist_matrix[i][j], 1)
        plt.plot(x, y, 'gray', linewidth=0.8)
        plt.text((x[0]+x[1])/2, (y[0]+y[1])/2, str(w), fontsize=7, color='darkred', ha='center')

    # Afficher les sommets
    for i, (x, y) in enumerate(coords):
        plt.scatter(x, y, c='blue')
        plt.text(x + 1, y + 1, str(i), fontsize=9)

    plt.title("Graphe TSP : sommets + arêtes pondérées")
    plt.grid(True)
    plt.axis('equal')
    plt.show()

    return coords, dist_matrix
