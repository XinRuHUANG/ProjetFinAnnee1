import itertools, math

# --- Données ---
D = [
    [0, 2, 9, 10],
    [2, 0, 6, 4],
    [9, 6, 0, 8],
    [10,4, 8, 0]
]
N = 4
NOEUDS   = range(N)
SOUS_ENS = [S for r in (2,3) for S in itertools.combinations(NOEUDS, r)]

# --- 1) init λ ---
def init_lambdas():
    return {S: 0.0 for S in SOUS_ENS}

# --- 2) coût modifié L(x,λ) sans la constante ---
def cout_modif(i,j, lambdas):
    # coût d_ij + pénalités pour chaque S contenant i et j
    return D[i][j] + sum(l for S,l in lambdas.items() if i in S and j in S)

# --- 3) résoudre le sous-problème assignation ---
def solve_assign(lambdas):
    best_val, best_perm, best_cnt = math.inf, None, None
    for perm in itertools.permutations(NOEUDS):
        if any(i==perm[i] for i in NOEUDS):  # interdit i→i
            continue
        var_cost = 0.0
        cnt = {S:0 for S in SOUS_ENS}
        for i,j in enumerate(perm):
            var_cost += cout_modif(i,j, lambdas)
            for S in SOUS_ENS:
                if i in S and j in S:
                    cnt[S] += 1
        const = -sum(l*(len(S)-1) for S,l in lambdas.items())
        total = var_cost + const
        if total < best_val:
            best_val, best_perm, best_cnt = total, perm, cnt
    return best_val, best_perm, best_cnt

# --- 4) mise à jour λ via sous-gradient simple ---
def maj_lambdas(lambdas, compte, pas):
    for S in SOUS_ENS:
        s = compte[S] - (len(S)-1)
        lambdas[S] = max(0.0, lambdas[S] + pas * s)

# --- 5) algorithme principal ---
def relaxation_lagrangienne(max_iter=20, pas0=2.0):
    lambdas = init_lambdas()
    for k in range(max_iter):
        LB, x_lambda, compte = solve_assign(lambdas)
        # si tu veux afficher LB à chaque itération
        print(f"it {k:2d}  LB={LB:.2f}")
        # critère de convergence sur le sous-gradient nul
        if all(compte[S]==len(S)-1 for S in SOUS_ENS):
            print("Sous-ensemble réalisable → x(lambda*) trouvé.")
            break
        pas = pas0/(k+1)
        maj_lambdas(lambdas, compte, pas)
    return LB, x_lambda, lambdas

if __name__ == "__main__":
    LB, x_star, lambda_star = relaxation_lagrangienne()
    print("\nRésultat final :")
    print("LB =", LB)
    print("x(lambda*) =", x_star)
    print("lambda* =", {S:round(l,2) for S,l in lambda_star.items()})
