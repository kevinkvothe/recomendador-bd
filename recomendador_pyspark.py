import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import findspark
findspark.init('/home/kubote/spark/spark-2.4.1-bin-hadoop2.7/')

import pyspark
from pyspark.sql import Row
from pyspark.ml.recommendation import ALS
try:
    sc.stop()
except:
    print("")


from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession

sc = SparkContext(master = "local[*]")
sqlContext = SQLContext(sc)
spark = SparkSession.builder.getOrCreate()


# Importamos los datos

# Leemos las filas del dataset una por una
lines = spark.read.text("./Datos/ratings.dat").rdd
lines2 = spark.read.text("./Datos/movies.dat").rdd

# Dividimos las filas en columnas.
parts = lines.map(lambda row: row.value.split("::"))
parts2 = lines2.map(lambda row: row.value.split("::"))

# Definimos cada variable con su tipo y nombre
ratingsRDD = parts.map(lambda p: Row(UserID = int(p[0]), MovieID = int(p[1]), Rating = float(p[2]), Timestamp = int(p[3])))
moviesRDD = parts2.map(lambda p: Row(MovieID = int(p[0]), Title = str(p[1]), Genres = str(p[2])))

# Lo transformamos en Dataframe.
ratings = spark.createDataFrame(ratingsRDD)
ratings = ratings.drop('Timestamp')

movies = spark.createDataFrame(moviesRDD)

# Resumen de los datos.
# ratings.describe().show()

# (training, test) = ratings.randomSplit([0.8, 0.2])

# Vamos a crear dos matrices a partir de los usuarios, los nombres de las películas y si han visto la película o no, a partir de esto realizaremos
# una recomendación por usuarios usando kneighbors y posteriormente una recomendación individual usando la correlación entre películas.

# Comenzamos creando la matriz de ratings. Sustituiremos los valores por 1 o 0 según han visto o no la película.

# Guardamos los valores únicos de películas y usuarios.
distinct_movies = ratings.select('MovieID').distinct().count()
distinct_users = ratings.select('UserID').distinct().count()

# Pivotamos sobre UserID y MovieID con la media (avg) de reating. Obtendremos una tabla con filas
# UserID, columnas MovieID y los valores medios de las valoraciones para cada combinación enmedio.
ratings_pivoted = ratings.groupBy("UserID").pivot("MovieID").avg("Rating")

# También creamos una tabla de contingencia que nos indique cuantas veces se produce la combinación
# de UserID y MovieID.
crossed = ratings.crosstab("UserID", "MovieID")

# Comprobamos que ambas tablas tengan las dimensiones correctas.
cierto = True if distinct_users == ratings_pivoted.count() else False
print("El número de filas de la tabla de ratings es correcto: ", cierto)

cierto = True if distinct_movies == len(ratings_pivoted.columns) - 1 else False
print("El número de columnas de la tabla de ratings es correcto: ", cierto)

cierto = True if distinct_users == crossed.count() else False
print("El número de filas de la tabla de contingencia es correcto: ", cierto)

cierto = True if distinct_movies == len(crossed.columns) - 1 else False
print("El número de columnas de la tabla de contingencia es correcto: ", cierto)

# Así pues, vamos a usar la tabla de contingencia para obtener un KMeans de 100 clústers,
# de esa forma encontraremos como pareja del seleccionado por una variable user_id al
# perteneciente a su mismo cluster.

# Definimos el user_id y el número de clusters
user_id = 1
n_clus = 100 # distinct_users / 2 if distinct_users % 2 == 0 else distinct_users / 2 - 1

# Transformamos la matriz para el KMeans.

# from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

vec_assembler = VectorAssembler(inputCols = crossed.columns[1:], outputCol='features')

kmeans_data = vec_assembler.transform(crossed)

# kmeans_data.show(5)

# Ajustamos el kmeans.

from pyspark.ml.clustering import KMeans
# from pyspark.ml.evaluation import ClusteringEvaluator


kmeans = KMeans(featuresCol='features', k = n_clus)
model = kmeans.fit(kmeans_data)

# Obtenemos las predicciones (esto añade una columna "prediction" al dataframe).
predictions = model.transform(kmeans_data)

# Buscamos el clúster al que pertenece nuestro usuario con UserID = 1.
predictions.filter(predictions.UserID_MovieID == 1).show()

cluster_id = predictions.filter(predictions.UserID_MovieID == 1).select('prediction').collect()[0]['prediction']
print("El clúster que buscamos es el :", cluster_id)

# Filtramos los usuarios pertenecientes a el cluster buscado y seleccionamos uno aleatorio.

possible_users = predictions.filter(predictions.prediction == cluster_id).filter(predictions.UserID_MovieID != 1).select('UserID_MovieID')
possible_users.show()

# Evaluamos el rendimiento

evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))








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
