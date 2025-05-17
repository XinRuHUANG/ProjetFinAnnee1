import numpy as np
from itertools import combinations
from scipy.optimize import linprog

class twoPhases:
    def __init__(self, MatriceDistance):
        self.dist = MatriceDistance
        self.n = len(MatriceDistance)
        self.subtourContraints = []
        
    def phase1_artificial_problem(self):
        # Nombre de variables (n*(n-1) variables de décision + 2n variables artificielles)
        nbVar = self.n * (self.n - 1)
        
        # Construction de la matrice de contraintes
        A_eq = np.zeros((2*self.n, nbVar))
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    var_idx = i*(self.n-1) + (j if j < i else j-1)
                    A_eq[i, var_idx] = 1  # Contraintes de départ
                    A_eq[self.n + j, var_idx] = 1  # Contraintes d'arrivée
        
        # Ajout des variables artificielles
        A_art = np.hstack([A_eq, np.eye(2*self.n)])
        b_eq = np.ones(2*self.n)
        c_art = np.zeros(nbVar + 2*self.n)
        c_art[nbVar:] = 1  # On minimise la somme des variables artificielles
        
        # Résolution avec linprog
        res = linprog(c=c_art, A_eq=A_art, b_eq=b_eq, bounds=(0, 1))
        
        if not res.success:
            raise ValueError("Phase 1 failed to find feasible solution")
            
        return res.x[:nbVar]
    
    def phase2_original_problem(self, initial_sol):
        solCourant = initial_sol.copy()
        iteration = 0
        iterMax = 100
        
        while iteration < iterMax:
            # Convertir en solution binaire (arrondir les valeurs)
            solBinaire = np.round(solCourant)
            
            # Détecter les sous-tours
            subtours = self.detect_subtours(solBinaire)
            if not subtours:
                return solBinaire
                
            # Ajouter des contraintes pour chaque sous-tour
            for subtour in subtours:
                self.add_subtour_constraint(subtour)
            
            # Réoptimiser avec les nouvelles contraintes
            solCourant = self.solve_with_constraints()
            iteration += 1
        
        raise ValueError("Maximum iterations reached without finding valid tour")
    
    def solve_with_constraints(self):
        nbVar = self.n * (self.n - 1)
        
        # Fonction objectif (minimiser la distance totale)
        c = np.zeros(nbVar)
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    idx = i*(self.n-1) + (j if j < i else j-1)
                    c[idx] = self.dist[i,j]
        
        # Contraintes de degré
        A_eq = np.zeros((2*self.n, nbVar))
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    var_idx = i*(self.n-1) + (j if j < i else j-1)
                    A_eq[i, var_idx] = 1  # Contraintes de départ
                    A_eq[self.n + j, var_idx] = 1  # Contraintes d'arrivée
        b_eq = np.ones(2*self.n)
        
        # Contraintes de sous-tours
        numSubtourContraints = len(self.subtourContraints)
        A_ub = np.zeros((numSubtourContraints, nbVar))
        b_ub = np.zeros(numSubtourContraints)
        
        for k, subtour in enumerate(self.subtourContraints):
            for i in subtour:
                for j in subtour:
                    if i != j:
                        idx = i*(self.n-1) + (j if j < i else j-1)
                        A_ub[k, idx] = 1
            b_ub[k] = len(subtour) - 1
        
        # Résolution
        res = linprog(c=c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=(0, 1))
        
        if not res.success:
            raise ValueError("Failed to solve with current constraints")
            
        return res.x
    
    def detect_subtours(self, solution):
        # Créer un graphe à partir de la solution
        graph = {i: [] for i in range(self.n)}
        for i in range(self.n):
            for j in range(self.n):
                if i != j and solution[i*(self.n-1) + (j if j < i else j-1)] > 0.5:
                    graph[i].append(j)
        
        # Détecter les cycles
        visited = [False] * self.n
        subtours = []
        
        for i in range(self.n):
            if not visited[i]:
                stack = [i]
                tour = []
                while stack:
                    node = stack.pop()
                    if visited[node]:
                        continue
                    visited[node] = True
                    tour.append(node)
                    for neighbor in graph[node]:
                        if not visited[neighbor]:
                            stack.append(neighbor)
                if len(tour) > 1 and len(tour) < self.n:
                    subtours.append(tour)
        
        return subtours
    
    def add_subtour_constraint(self, subtour):
        # Ajouter une contrainte pour éliminer ce sous-tour
        # ∑_{i,j ∈ subtour} x_ij ≤ |subtour| - 1
        self.subtourContraints.append(subtour)
    
    def solution_to_tour(self, solution):
        # Reconstruire le tour à partir de la solution binaire
        graph = {i: [] for i in range(self.n)}
        for i in range(self.n):
            for j in range(self.n):
                if i != j and solution[i*(self.n-1) + (j if j < i else j-1)] > 0.5:
                    graph[i].append(j)
        
        tour = [0]
        current = 0
        visited = set([0])
        
        while len(visited) < self.n:
            next_city = graph[current][0]  # Prendre le seul successeur
            if next_city in visited and len(visited) < self.n:
                # Trouver une autre arête si on crée un cycle prématuré
                for j in range(self.n):
                    if j != current and solution[current*(self.n-1) + (j if j < current else j-1)] > 0.5 and j not in visited:
                        next_city = j
                        break
            tour.append(next_city)
            visited.add(next_city)
            current = next_city
        
        return tour
    
    def solve(self):
        # Phase 1: Trouver une solution réalisable initiale
        print("Running Phase 1...")
        initial_sol = self.phase1_artificial_problem()
        
        # Phase 2: Optimiser avec élimination des sous-tours
        print("Running Phase 2...")
        solution = self.phase2_original_problem(initial_sol)
        
        return solution

# Matrice de distance
MatriceDistance = np.array([
    [0, 2, 9, 10],
    [2, 0, 6, 4],
    [9, 6, 0, 8],
    [10, 4, 8, 0]
])

# Création et résolution du solveur
print("Initializing solver...")
solver = twoPhases(MatriceDistance)
print("Solving TSP...")
solution = solver.solve()

# Affichage des résultats
print("\nSolution finale:")
for i in range(solver.n):
    for j in range(solver.n):
        if i != j and solution[i*(solver.n-1) + (j if j < i else j-1)] > 0.5:
            print(f"Ville {i} → Ville {j} (distance: {MatriceDistance[i,j]})")

# Conversion en tour
tour = solver.solution_to_tour(solution)
distanceTotal = sum(MatriceDistance[tour[i]][tour[i+1]] for i in range(len(tour)-1))
distanceTotal += MatriceDistance[tour[-1]][tour[0]]  # Retour au point de départ

print("\nTour optimal:", tour + [tour[0]])
print(f"Distance totale: {distanceTotal}")