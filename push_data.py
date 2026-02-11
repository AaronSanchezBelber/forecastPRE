# Importamos el módulo os para manejar variables de entorno
import os

# Importamos pandas para la manipulación de datos
import pandas as pd

# Importamos pymongo para interactuar con MongoDB
import pymongo

# Importamos load_dotenv para cargar variables desde el archivo .env
from dotenv import load_dotenv

# Importamos certifi para manejar certificados SSL
import certifi

# Cargamos las variables de entorno desde el archivo .env
load_dotenv()

# Obtenemos la URL de conexión a MongoDB desde las variables de entorno
MONGO_DB_URL = os.getenv("MONGO_DB_URL")

# Obtenemos la ruta del certificado SSL
ca = certifi.where()


# Definimos la clase DataPusher
class DataPusher:
    def __init__(self):
        # Constructor de la clase (no inicializa atributos por ahora)
        pass

    def csv_to_json_convertor(self, file_path):
        """
        Lee un archivo CSV y lo convierte en una lista de diccionarios
        lista para ser insertada en MongoDB
        """
        try:
            # Leemos el archivo CSV usando pandas
            data = pd.read_csv(file_path)

            # Reseteamos el índice del DataFrame
            data.reset_index(drop=True, inplace=True)

            # Eliminamos la columna de índice autogenerada si existe
            if "Unnamed: 0" in data.columns:
                data = data.drop(columns=["Unnamed: 0"])

            # Convertimos la columna 'date' a tipo datetime si existe
            # Esto es importante para la validación del esquema en MongoDB
            if "date" in data.columns:
                data["date"] = pd.to_datetime(data["date"], errors="coerce")

            # Convertimos el DataFrame a una lista de diccionarios
            # orient="records" preserva los objetos datetime para BSON
            records = data.to_dict(orient="records")

            return records
        except Exception as e:
            # Lanzamos un error personalizado si falla la lectura del CSV
            raise RuntimeError(f"Error leyendo CSV: {e}")

    def pushing_data_to_mongodb(self, records, database, collection):
        """
        Inserta los registros en la base de datos y colección indicadas
        """
        try:
            # Verificamos que la URL de MongoDB esté definida
            if not MONGO_DB_URL:
                raise ValueError("MONGO_DB_URL no está definido en el entorno.")

            # Creamos el cliente de MongoDB usando SSL
            mongo_client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=ca)

            # Seleccionamos la base de datos
            db = mongo_client[database]

            # Seleccionamos la colección
            col = db[collection]

            # Insertamos múltiples documentos en la colección
            result = col.insert_many(records)

            # Retornamos el número de documentos insertados
            return len(result.inserted_ids)
        except Exception as e:
            # Lanzamos un error personalizado si falla la inserción
            raise RuntimeError(f"Error insertando en MongoDB: {e}")


# Punto de entrada principal del script
if __name__ == "__main__":
    # Ruta del archivo CSV
    FILE_PATH = "./data/sales_train_merged_.csv"

    # Nombre de la base de datos
    DATABASE = "SalesForecast2026"

    # Nombre de la colección
    COLLECTION = "forecast"

    # Creamos una instancia de DataPusher
    pusher = DataPusher()

    # Convertimos el CSV a registros tipo JSON/dict
    records = pusher.csv_to_json_convertor(FILE_PATH)

    # Insertamos los registros en MongoDB
    noofrecords = pusher.pushing_data_to_mongodb(records, DATABASE, COLLECTION)

    # Mostramos en consola cuántos registros se insertaron
    print(noofrecords)
