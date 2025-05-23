import numpy as np

def construire_tableau_tsp(c):
    n = c.shape[0]
    # Variables x_ij pour i != j
    # nombre de variables : n*(n-1)
    variables = [(i,j) for i in range(n) for j in range(n) if i != j]
    num_vars = len(variables)
    
    contraintes = []
    b = []
    
    # Contraintes de sortie : pour chaque ville i, somme x_ij = 1
    for i in range(n):
        coef = np.zeros(num_vars)
        for idx, (a,b_) in enumerate(variables):
            if a == i:
                coef[idx] = 1
        contraintes.append(coef)
        b.append(1)
    
    # Contraintes d'entrée : pour chaque ville j, somme x_ij =1
    for j in range(n):
        coef = np.zeros(num_vars)
        for idx, (a,b_) in enumerate(variables):
            if b_ == j:
                coef[idx] = 1
        contraintes.append(coef)
        b.append(1)
    
    A = np.vstack(contraintes)
    b = np.array(b)
    c_vec = np.array([c[i,j] for i,j in variables])
    
    # Toutes contraintes sont des égalités
    contraintes_types = ['=']*len(b)
    
    return A, b, c_vec, contraintes_types, variables

# Appel de la méthode des 2 phases (celle que j’ai fournie précédemment) sur ce problème

# Exemple d'utilisation (distance euclidienne entre points)
def exemple():
    # Matrice de coûts (distances) données directement
    c = np.array([
        [0, 2, 9, 10],
        [2, 0, 6, 4],
        [9, 6, 0, 8],
        [10, 4, 8, 0]
    ], dtype=float)
    
    n = c.shape[0]
    
    # Interdit les arcs de i vers i
    for i in range(n):
        c[i,i] = 1e6  # très grand coût pour empêcher boucle sur soi-même
    
    A, b, c_vec, contraintes, variables = construire_tableau_tsp(c)
    
    from DeuxPhases import methode_deux_phases  # ta fonction des 2 phases
    
    solution, val = methode_deux_phases(A, b, c_vec, contraintes, objectif='min')
    print("Solution relaxation LP TSP :", solution)
    print("Valeur minimale (relaxation):", val)


exemple()
