def two_opt_tour(tour, dist_matrix):
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for i in range(1, n-2):
            for j in range(i+1, n):
                if j-i == 1: continue
                # Coût actuel : d(i,i+1) + d(j,j+1)
                current = dist_matrix[tour[i-1]][tour[i]] + dist_matrix[tour[j-1]][tour[j]]
                # Coût après échange : d(i,j) + d(i+1,j+1)
                new = dist_matrix[tour[i-1]][tour[j-1]] + dist_matrix[tour[i]][tour[j]]
                if new < current:
                    # Inverser la sous-séquence entre i et j
                    tour[i:j] = tour[j-1:i-1:-1]
                    improved = True
    return tour

def two_opt_tsp(dist_matrix, initial_tour=None):
    n = len(dist_matrix)
    if initial_tour is None:
        tour = list(range(n))  # Tour initial trivial
    else:
        tour = initial_tour.copy()
    
    tour = two_opt_tour(tour, dist_matrix)
    # Calcul du coût total
    cout = sum(dist_matrix[tour[i]][tour[(i+1)%n]] for i in range(n))
    return tour, cout