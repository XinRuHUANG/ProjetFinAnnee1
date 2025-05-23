import numpy as np
from scipy.optimize import linprog

def executerPhase1(matriceDistances):
    nbVilles = len(matriceDistances)
    nbVariables = nbVilles * (nbVilles - 1)
    
    matriceContraintesDegre = np.zeros((2 * nbVilles, nbVariables))
    
    for villeDep in range(nbVilles):
        for villeArr in range(nbVilles):
            if villeDep != villeArr:
                idxVariable = villeDep * (nbVilles - 1) + (villeArr if villeArr < villeDep else villeArr - 1)
                matriceContraintesDegre[villeDep, idxVariable] = 1
                matriceContraintesDegre[nbVilles + villeArr, idxVariable] = 1
    
    matriceAvecVariablesArtificielles = np.hstack([matriceContraintesDegre, np.eye(2 * nbVilles)])
    coutVariablesArtificielles = np.zeros(nbVariables + 2 * nbVilles)
    coutVariablesArtificielles[nbVariables:] = 1
    
    resultat = linprog(
        c=coutVariablesArtificielles,
        A_eq=matriceAvecVariablesArtificielles,
        b_eq=np.ones(2 * nbVilles),
        bounds=(0, 1),
        method='highs'
    )
    
    if not resultat.success:
        raise ValueError("Aucune solution réalisable trouvée en phase 1")
        
    return resultat.x[:nbVariables]

def executerPhase2(matriceDistances, solutionInitiale):
    nbVilles = len(matriceDistances)
    solutionActuelle = solutionInitiale.copy()
    listeContraintesSousTours = []
    nbIterationsMax = 100
    
    for _ in range(nbIterationsMax):
        solutionBinaire = np.round(solutionActuelle)
        listeSousTours = detecterSousTours(solutionBinaire, nbVilles)
        
        if not listeSousTours:
            return solutionBinaire
            
        for sousTour in listeSousTours:
            listeContraintesSousTours.append(sousTour)
        
        solutionActuelle = resoudreAvecContraintes(matriceDistances, listeContraintesSousTours)
    
    raise ValueError("Nombre maximal d'itérations atteint")

def resoudreAvecContraintes(matriceDistances, listeContraintesSousTours):
    nbVilles = len(matriceDistances)
    nbVariables = nbVilles * (nbVilles - 1)
    
    coutDistances = np.zeros(nbVariables)
    for villeDep in range(nbVilles):
        for villeArr in range(nbVilles):
            if villeDep != villeArr:
                idxVariable = villeDep * (nbVilles - 1) + (villeArr if villeArr < villeDep else villeArr - 1)
                coutDistances[idxVariable] = matriceDistances[villeDep, villeArr]
    
    matriceContraintesDegre = np.zeros((2 * nbVilles, nbVariables))
    for villeDep in range(nbVilles):
        for villeArr in range(nbVilles):
            if villeDep != villeArr:
                idxVariable = villeDep * (nbVilles - 1) + (villeArr if villeArr < villeDep else villeArr - 1)
                matriceContraintesDegre[villeDep, idxVariable] = 1
                matriceContraintesDegre[nbVilles + villeArr, idxVariable] = 1
    
    nbContraintes = len(listeContraintesSousTours)
    matriceContraintesSousTours = np.zeros((nbContraintes, nbVariables))
    bornesContraintes = np.zeros(nbContraintes)
    
    for idxContrainte, sousTour in enumerate(listeContraintesSousTours):
        for villeDep in sousTour:
            for villeArr in sousTour:
                if villeDep != villeArr:
                    idxVariable = villeDep * (nbVilles - 1) + (villeArr if villeArr < villeDep else villeArr - 1)
                    matriceContraintesSousTours[idxContrainte, idxVariable] = 1
        
        bornesContraintes[idxContrainte] = len(sousTour) - 1
    
    resultat = linprog(
        c=coutDistances,
        A_eq=matriceContraintesDegre,
        b_eq=np.ones(2 * nbVilles),
        A_ub=matriceContraintesSousTours,
        b_ub=bornesContraintes,
        bounds=(0, 1),
        method='highs'
    )
    
    if not resultat.success:
        raise ValueError("Échec de la résolution")
        
    return resultat.x

def detecterSousTours(solution, nbVilles):
    grapheSolution = {ville: [] for ville in range(nbVilles)}
    
    for villeDep in range(nbVilles):
        for villeArr in range(nbVilles):
            if villeDep != villeArr:
                idxVariable = villeDep * (nbVilles - 1) + (villeArr if villeArr < villeDep else villeArr - 1)
                if solution[idxVariable] > 0.5:
                    grapheSolution[villeDep].append(villeArr)
    
    villesVisitees = [False] * nbVilles
    listeSousTours = []
    
    for villeCourante in range(nbVilles):
        if not villesVisitees[villeCourante]:
            pile = [villeCourante]
            sousTourCourant = []
            
            while pile:
                noeud = pile.pop()
                if villesVisitees[noeud]:
                    continue
                
                villesVisitees[noeud] = True
                sousTourCourant.append(noeud)
                
                for voisin in grapheSolution[noeud]:
                    if not villesVisitees[voisin]:
                        pile.append(voisin)
            
            if 1 < len(sousTourCourant) < nbVilles:
                listeSousTours.append(sousTourCourant)
    
    return listeSousTours

def convertirEnTour(solution, matriceDistances):
    nbVilles = len(matriceDistances)
    grapheSolution = {ville: [] for ville in range(nbVilles)}
    
    for villeDep in range(nbVilles):
        for villeArr in range(nbVilles):
            if villeDep != villeArr:
                idxVariable = villeDep * (nbVilles - 1) + (villeArr if villeArr < villeDep else villeArr - 1)
                if solution[idxVariable] > 0.5:
                    grapheSolution[villeDep].append(villeArr)
    
    tour = [0]
    villeCourante = 0
    villesVisitees = {0}
    
    while len(villesVisitees) < nbVilles:
        for villeSuivante in grapheSolution[villeCourante]:
            if villeSuivante not in villesVisitees:
                tour.append(villeSuivante)
                villesVisitees.add(villeSuivante)
                villeCourante = villeSuivante
                break
    
    return tour

def resoudreTSP(matriceDistances):
    print("Phase 1: Recherche d'une solution initiale réalisable...")
    solutionPhase1 = executerPhase1(matriceDistances)
    
    print("Phase 2: Optimisation et élimination des sous-tours...")
    solutionFinale = executerPhase2(matriceDistances, solutionPhase1)
    
    tourComplet = convertirEnTour(solutionFinale, matriceDistances)
    distanceTotale = sum(matriceDistances[tourComplet[i]][tourComplet[i+1]] 
                       for i in range(len(tourComplet)-1))
    distanceTotale += matriceDistances[tourComplet[-1]][tourComplet[0]]
    
    return solutionFinale, tourComplet + [tourComplet[0]], distanceTotale

if __name__ == "__main__":
    distancesVilles = np.array([
        [0, 2, 9, 10],
        [2, 0, 6, 4],
        [9, 6, 0, 8],
        [10, 4, 8, 0]
    ])
    
    print("Résolution du TSP en cours...")
    solutionOptimale, tourComplet, distanceTotale = resoudreTSP(distancesVilles)
    nbVilles = len(distancesVilles)
    
    print("\nArêtes sélectionnées dans la solution:")
    for villeDep in range(nbVilles):
        for villeArr in range(nbVilles):
            if villeDep != villeArr:
                idxVariable = villeDep * (nbVilles - 1) + (villeArr if villeArr < villeDep else villeArr - 1)
                if solutionOptimale[idxVariable] > 0.5:
                    print(f"Ville {villeDep} → Ville {villeArr} (distance: {distancesVilles[villeDep, villeArr]})")
    
    print("\nTour optimal trouvé:", tourComplet)
    print(f"Distance totale du tour: {distanceTotale}")
