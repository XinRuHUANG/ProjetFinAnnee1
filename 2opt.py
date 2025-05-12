def twoOpt(tour, dist_matrix):
    n = len(tour)
    test = True
    while test:
        test = False
        for i in range(1, n-2):
            for j in range(i+1, n):
                if j-i == 1: continue
                # Coût actuel : d(i,i+1) + d(j,j+1)
                cout = dist_matrix[tour[i-1]][tour[i]] + dist_matrix[tour[j-1]][tour[j]]
                # Coût après échange : d(i,j) + d(i+1,j+1)
                coutNouveau = dist_matrix[tour[i-1]][tour[j-1]] + dist_matrix[tour[i]][tour[j]]
                if coutNouveau < cout:
                    # Inverser la sous-séquence entre i et j
                    tour[i:j] = tour[j-1:i-1:-1]
                    test = True
    return tour

def twoOptPVC(dist_matrix, tourInitial=None):
    n = len(dist_matrix)
    if tourInitial is None:
        tour = list(range(n))  # Tour initial trivial
    else:
        tour = tourInitial.copy()
    
    tour = twoOpt(tour, dist_matrix)
    # Calcul du coût total
    cout = sum(dist_matrix[tour[i]][tour[(i+1)%n]] for i in range(n))
    return tour, cout

