# Sistema de Recomendacion basado en Filtrado Colaborativo
# Miguel Lozano: 2019

from sklearn.neighbors import NearestNeighbors
import numpy as np

NP = 100     # Number of Products
NU = 100     # Number of users

# Creamos las matrices de usuarios-producto (inventory) y de similaridad entre productos (similarity)
inventory = np.random.randint(2, size = (NU, NP))
similarity = np.random.rand(NP,NP)

for i in range(NP):
  for j in range(i, NP):
      similarity[j,i] = similarity[i,j]
  similarity[i,i] = 1

print(inventory)
print(similarity)

# Calculamos los vecinos mas proximos (3) al user_id
user_id = 0
nbrs = NearestNeighbors(n_neighbors=3).fit(inventory)
distances, indices = nbrs.kneighbors(inventory)

print("Distancias y vecinos: ", distances[user_id], indices[user_id])

# Criterio de recomendacion: el mas similar al preferido por el usuario y no escogido todavia
# Primero calculamos las diferencia entre los productos comprados por el user_id  y su vecino

# Calculamos las diferencias con todos los vecinos primero
difs = inventory - inventory[user_id]
vecino = indices[user_id][1]   # 2ndo vecino, el primero es el mismo
print(vecino, difs[vecino])    # los 1's son los posibles productos a recomendar: comprados por vecino y no por user_id
posibles = (difs[vecino]  == 1)
Lposibles = np.asarray(np.where(posibles == True))[0]   # Lista de posibles productos a recomedar, sin valorar

#print("Similarity: \n", similarity)
# Cual de ellos es el mejor ... el que mas similar a los productos que compra user_id
# Valoracion de los posibles productos en base a la similitud con los que compra user_id
if len(Lposibles > 0):
    Lscores = []
    for p in Lposibles:
        # score p
        score = 0

        for i in range(len(inventory[user_id])):
            if inventory[user_id][i] == 1:
                score += similarity[i][p]
        Lscores.append(score)

        # Está mal, no valora la realidad.
        # for up in inventory[user_id]:
        #     if up == 1:
        #         score += similarity[up][p]
        # Lscores.append(score)
    # Similitud del primer producto posible con todos los comprados por user_id
    # similarity[np.where(inventory[user_id] == 1)][:, Lposibles[0]].sum()

    print("Posibles productos: ", Lposibles)
    print("Scores: ", Lscores)
    irecom = int(np.array(Lscores).argmax())
    print("Recomendacion Final: ", irecom, Lposibles[irecom]) 
