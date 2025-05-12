# coding=utf-8
import numpy as np # type: ignore

def simplexe(A, b, c, permut, max_iter=100, tol=1e-10, verbose=False):
    """
    Méthode du simplexe pour résoudre :
        max c^T x  sous les contraintes Ax = b, x >= 0

    Arguments :
        A : matrice des contraintes (m x n)
        b : vecteur des contraintes (longueur m)
        c : vecteur des coefficients objectifs (longueur n)
        permut : permutation initiale des variables (base d'abord, puis hors-base)
        max_iter : nombre maximum d'itérations
        tol : tolérance pour test d'optimalité
        verbose : affiche les échanges base/hors-base à chaque étape

    Retourne :
        x : solution optimale
        gain : valeur maximale de la fonction objectif
        ou un message d'erreur si le problème est mal posé
    """
    m, n = A.shape

    if m >= n or c.shape[0] != n or b.shape[0] != m:
        return 'Dimensions incompatibles'

    if np.any(b < 0):
        return 'Le vecteur b doit être >= 0'

    for _ in range(max_iter):
        # Ap : matrice A avec colonnes permutées
        Ap = A[:, permut]
        cp = c[permut]

        # Test d'inversibilité
        base_matrix = Ap[:, :m]
        if np.linalg.matrix_rank(base_matrix) < m:
            return 'Matrice non inversible (base non admissible)'

        invAp = np.linalg.inv(base_matrix)
        Chb = np.dot(invAp, Ap[:, m:])
        bbase = np.dot(invAp, b)

        # Coefficients du gain pour les variables hors base
        cbase = -np.dot(cp[:m], Chb) + cp[m:]

        # Test d’optimalité
        if np.all(cbase <= tol):
            break  # Optimum atteint

        # Variable qui entre dans la base (celle qui maximise le gain)
        ihb = np.argmax(cbase) + m
        direction = Chb[:, ihb - m]

        # Test de non borné
        if np.all(direction <= tol):
            return 'Problème non borné'

        # Calcul des ratios pour déterminer la variable qui sort
        ratios = np.array([
            bbase[i] / direction[i] if direction[i] > tol else np.inf
            for i in range(m)
        ])
        ib = np.argmin(ratios)

        if verbose:
            print(f"out = x{permut[ib]}   in = x{permut[ihb]}")

        # Échange des variables base/hors base
        permut[ib], permut[ihb] = permut[ihb], permut[ib]
    else:
        return 'Non convergence après le nombre maximal d’itérations'

    # Construction de la solution finale
    xp = np.hstack((bbase, np.zeros(n - m)))
    x = np.empty(n)
    for i in range(n):
        x[permut[i]] = xp[i]

    gain = float(np.dot(c, x))
    return x, gain

# Exemple issu du cours
A = np.array([
    [1, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 0],
    [3, 6, 2, 0, 0, 0, 1]
])
b = np.array([1000, 500, 1500, 6750])
c = np.array([4, 12, 3, 0, 0, 0, 0])
permut = np.array([6, 5, 4, 3, 2, 1, 0])  # base initiale : colonnes 6 à 3

solution = simplexe(A, b, c, permut, verbose=True)
print(solution)
