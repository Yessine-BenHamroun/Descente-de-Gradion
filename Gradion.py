import numpy as np

# Positions des utilisateurs (exemple)
U = np.array([[1, 2], [3, 5], [6, 8], [7, 3]])

# Initialisation des positions des antennes (aléatoire)
X = np.random.rand(2, 2) * 10  # 2 antennes avec des coordonnées aléatoires dans un plan de 10x10

# Fonction de calcul de la distance entre antennes et utilisateurs
def distance(x, u):
    return np.linalg.norm(x - u, axis=1)

# Fonction coût (somme des distances minimales)
def cost(X, U):
    total_cost = 0
    for u in U:
        total_cost += np.min(distance(X, u))
    return total_cost

# Paramètres de la descente de gradient
alpha = 0.01
max_iter = 1000

# Boucle de descente de gradient
for it in range(max_iter):
    grad = np.zeros_like(X)
    
    # Calcul du gradient pour chaque antenne
    for i in range(len(X)):
        for u in U:
            dist = distance(X[i], u)
            if dist > 0:  # Éviter les divisions par zéro
                grad[i] += (X[i] - u) / dist
    
    # Mise à jour des positions des antennes
    X -= alpha * grad
    
    # Affichage du coût actuel (facultatif)
    if it % 100 == 0:
        print(f"Itération {it}, Coût: {cost(X, U)}")

# Affichage des positions finales des antennes
print("Positions optimisées des antennes :", X)
