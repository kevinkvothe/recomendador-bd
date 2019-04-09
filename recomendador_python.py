
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Importamos los datos
ratings = pd.read_csv("./Datos/ratings.dat", sep = "::", header = None, names = ["UserID", "MovieID", "Rating", "Timestamp"])
movies = pd.read_csv("./Datos/movies.dat", sep = "::", header = None, names = ["MovieID", "Title", "Genres"])

# Vamos a crear dos matrices a partir de los usuarios, los nombres de las películas y si han visto la película o no, a partir de esto realizaremos
# una recomendación por usuarios usando kneighbors y posteriormente una recomendación individual usando la correlación entre películas.

# Comenzamos creando la matriz de ratings.
ratings_matrix = ratings.pivot_table(index = "UserID", columns = "MovieID", values = "Rating")
ratings_matrix_bin = np.where(np.isnan(ratings_matrix), 0, 1)

ratings_matrix_cop = pd.DataFrame(ratings_matrix)
movie_id = ratings_matrix_cop.columns.values
users_id = ratings_matrix_cop.index
# ratings_matrix = np.matrix(ratings_matrix)

# Calculamos los vecinos mas proximos (3) al user_id (0 por defecto).
user_id = 0
nbrs = NearestNeighbors(n_neighbors=3).fit(ratings_matrix_bin)
distances, indices = nbrs.kneighbors([ratings_matrix_bin[user_id]])

print("Distancias y vecinos al usuario con UserID = ", user_id, ":",  distances, indices)

# Criterio de recomendacion: el mas similar al preferido por el usuario y no escogido todavia
# Primero calculamos las diferencia entre los productos comprados por el user_id  y su vecino

# Calculamos las diferencias con todos los vecinos primero
difs = ratings_matrix_bin - ratings_matrix_bin[user_id]
vecino = indices[0, 1]   # 2ndo vecino, el primero es el mismo
print(vecino, difs[vecino])   # los 1's son los posibles productos a recomendar: comprados por vecino y no por user_id
posibles = (difs[vecino]  == 1)
Lposibles = movie_id[posibles]  # Lista de posibles películas a recomedar, sin valorar.

# Observamos las películas vistas por el user_id.
vistas = movie_id[np.where(ratings_matrix_bin[user_id] == 1)[0]]

print("Películas vistas por el usuario: ", users_id[user_id])
for i in vistas:
    print(movies.iloc[np.where(movies.MovieID == i)[0], :])

print("Películas Posibles: ")
for i in Lposibles:
    print(movies.iloc[np.where(movies.MovieID == i)[0], :])

# Ahora vamos a trabajar con la matriz ratings_matrix, buscando la correlación entre películas a partir de las valoraciones de usuarios.
if len(Lposibles > 0):

    Lscores = []

    for p in Lposibles:

        # Guardamos la matriz de ratings de la p ésima película a recomendar.
        ratings_matrix_pos = ratings_matrix[p]

        # Guardamos la matriz de ratings de las películas vistas por el usuario UserID.
        ratings_matrix_user = ratings_matrix[vistas]

        # Calculamos la similitud entre la película posible y las vistas por el usuario.
        simil = pd.DataFrame(ratings_matrix_user.corrwith(ratings_matrix_pos))

        # Almacenamos la suma de las correlaciones de la posible película con las vistas.
        Lscores.append(simil.sum()[0])

    irecom = int(np.array(Lscores).argmax())
    print("Recomendacion Final: ")
    print(movies.iloc[np.where(movies.MovieID == Lposibles[irecom])[0], :])
