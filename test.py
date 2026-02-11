# Importamos el cliente de MongoDB desde pymongo
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# Importamos load_dotenv para cargar variables de entorno desde un archivo .env
from dotenv import load_dotenv
import os

# Cargamos las variables de entorno definidas en el archivo .env
load_dotenv()

# Obtenemos la URL de conexión a MongoDB desde la variable de entorno
MONGO_DB_URL = os.getenv("MONGO_DB_URL")

# Asignamos la URI de conexión
uri = MONGO_DB_URL

# Creamos un nuevo cliente de MongoDB y nos conectamos al servidor
client = MongoClient(uri, server_api=ServerApi('1'))

# Enviamos un comando 'ping' para confirmar que la conexión fue exitosa
try:
    client.admin.command('ping')
    print("Ping enviado correctamente. ¡Conexión exitosa a MongoDB!")
except Exception as e:
    # Si ocurre algún error, lo mostramos en consola
    print(e)
