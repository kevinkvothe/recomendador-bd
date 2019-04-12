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
    print("Stopped and restarted")
except:
    print("Nothing to stop")

from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession

sc = SparkContext(master = "local[*]")
sqlContext = SQLContext(sc)
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

# ratings_pivoted.corr('110', '2')

# También creamos una tabla de contingencia que nos indique cuantas veces se produce la combinación
# de UserID y MovieID.
crossed = ratings.crosstab("UserID", "MovieID")

# Necesitaremos los id de los usuarios posteriormente.
users_total_id = [int(row['UserID_MovieID']) for row in crossed.select('UserID_MovieID').collect()]

# Comprobamos que ambas tablas tengan las dimensiones correctas.
# cierto = True if distinct_users == ratings_pivoted.count() else False
# print("El número de filas de la tabla de ratings es correcto: ", cierto)

# cierto = True if distinct_movies == len(ratings_pivoted.columns) - 1 else False
# print("El número de columnas de la tabla de ratings es correcto: ", cierto)

# cierto = True if distinct_users == crossed.count() else False
# print("El número de filas de la tabla de contingencia es correcto: ", cierto)

# cierto = True if distinct_movies == len(crossed.columns) - 1 else False
# print("El número de columnas de la tabla de contingencia es correcto: ", cierto)

# Así pues, vamos a usar la tabla de contingencia para obtener un KMeans de 100 clústers,
# de esa forma encontraremos como pareja del seleccionado por una variable user_id al
# perteneciente a su mismo cluster.

# Definimos el user_id y el número de clusters
user_id_position = 1
n_clus = 100 # distinct_users / 2 if distinct_users % 2 == 0 else distinct_users / 2 - 1

# Transformamos la matriz para el KMeans.
# from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

vec_assembler = VectorAssembler(inputCols = crossed.columns[1:], outputCol='features')
kmeans_data = vec_assembler.transform(crossed)

# Ajustamos el kmeans.
from pyspark.ml.clustering import KMeans

kmeans = KMeans(featuresCol='features', k = n_clus)
model = kmeans.fit(kmeans_data)

# Obtenemos las predicciones (esto añade una columna "prediction" al dataframe).
predictions = model.transform(kmeans_data)

# Buscamos el clúster al que pertenece nuestro usuario en la posición 1 (0 en python, el primero).
user_id = users_total_id[user_id_position]
# predictions.filter(predictions.UserID_MovieID == user_id).show()

cluster_id = predictions.filter(predictions.UserID_MovieID == 1).select('prediction').collect()[0]['prediction']
print("El clúster que buscamos es el :", cluster_id)

# Filtramos los usuarios pertenecientes a el cluster buscado y seleccionamos uno aleatorio.
possible_users = predictions.filter(predictions.prediction == cluster_id).filter(predictions.UserID_MovieID != 1).select('UserID_MovieID')

# Seleccionamos uno aleatorio a partir del cual recomendar películas a user_id.
similar_user = np.random.randint(possible_users.count())
similar_user_movies = predictions.filter(predictions.UserID_MovieID == possible_users.collect()[similar_user]['UserID_MovieID'])

# Calculamos las diferencias con el vector del usuario seleccionado con el similar. Primero definimos dos arrays.
similar_user_movies_array = np.array(similar_user_movies.drop("UserID_MovieID", "features", "prediction").collect())
user_movies_array = np.array(predictions.filter(predictions.UserID_MovieID == 1).drop("UserID_MovieID", "features", "prediction").collect())

# Y ahora la diferencia.
diff = similar_user_movies_array - user_movies_array

# Los códigos de las películas.
id_peliculas = np.array(similar_user_movies.drop("UserID_MovieID", "features", "prediction").columns)

# Ahora, las posiciones de las posibles películas (las que tengan valor 1).
Lposibles = id_peliculas[np.where(diff == 1)[1]]

# Y las películas vistas por el usuario user_id.
vistas_user = id_peliculas[np.where(user_movies_array == 1)[1]]

# Mostramos las películas vistas y posibles.
print("Películas vistas por el usuario: ", user_id)
print(vistas_user)

print("Películas Posibles: ")
print(Lposibles)

# ratings_pivoted_filled = ratings_pivoted.fillna(0)

# Ahora simplemente buscaremos de las películas posibles la de mayor media.

from pyspark.sql.functions import mean
medias = ratings.groupby('MovieID').agg({'Rating': "mean"})

medias_movies_id = np.array([int(row['MovieID']) for row in medias.select('MovieID').collect()])
medias_movies_avg = [float(row['avg(Rating)']) for row in medias.select('avg(Rating)').collect()]

if Lposibles.shape[0] > 0:

    Lscores = []
    i = 0

    for p in Lposibles:

        media = medias_movies_avg[np.where(medias_movies_id == int(p))[0][0]]
        # media = ratings_pivoted.select(p).dropna().select(mean(p)).collect()[0]['avg(' + p + ')']

        # Almacenamos la suma de las correlaciones de la posible película con las vistas.
        Lscores.append(media)


    irecom = int(np.array(Lscores).argmax())
    print("Recomendacion Final: ")
    print(movies.filter(movies.MovieID == Lposibles[irecom]).select('Title').show())












import pyspark.sql.functions as func

def cosine_similarity(df, col1, col2):

    df_cosine = df.select(func.sum(df[col1] * df[col2]).alias('dot'),
                          func.sqrt(func.sum(df[col1]**2)).alias('norm1'),
                          func.sqrt(func.sum(df[col2] **2)).alias('norm2'))

    d = df_cosine.rdd.collect()[0].asDict()

    return d['dot']/(d['norm1'] * d['norm2'])

def euclidean_distance(df, col1, col2):

    df_cosine = df.select(func.sqrt(func.sum((df[col1] - df[col2])**2)).alias('dot'))

    d = df_cosine.rdd.collect()[0].asDict()

    return d['dot']

# euclidean_distance(ratings_pivoted, '110', '1')

# Ahora vamos a trabajar con la matriz ratings_pivoted, buscando la correlación entre películas a partir de las valoraciones de usuarios.
if Lposibles.shape[0] > 0:

    Lscores = []
    i = 0

    for p in Lposibles:

        simil = 0

        for u in vistas_user:

            print("Iteración ", i, " de ", Lposibles.shape[0]*vistas_user.shape[0])
            i = i + 1

            # Calculamos la similitud entre la película posible y las vistas por el usuario.
            simil = simil + euclidean_distance(ratings_pivoted, p, u) #ratings_pivoted_filled.corr(p, u)

        # Almacenamos la suma de las correlaciones de la posible película con las vistas.
        Lscores.append(simil)


    irecom = int(np.array(Lscores).argmin())
    print("Recomendacion Final: ")
    print(movies.filter(movies.MovieID == Lposibles[irecom]).select('Title').show())
