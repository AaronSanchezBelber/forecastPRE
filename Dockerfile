# 1️⃣ Portabilidad
# 2️⃣ Reproducibilidad
# 3️⃣ Escalabilidad
# 4️⃣ Base de CI/CD moderno
# 5️⃣ Aislamiento

# Imagen base oficial de Python ligera (Debian slim)
# Más pequeña que la imagen completa → mejor para producción
FROM python:3.11-slim

# Variables de entorno:
# PYTHONDONTWRITEBYTECODE=1 → evita que Python genere archivos .pyc
# PYTHONUNBUFFERED=1 → fuerza que los logs salgan inmediatamente (útil en Docker y Cloud Run)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1


# Define el directorio de trabajo dentro del contenedor
# Todo lo que ocurra después se ejecuta dentro de /app
WORKDIR /app

# Instala dependencias del sistema necesarias para compilar librerías Python
# build-essential incluye gcc y herramientas de compilación
# Luego limpia cache de apt para reducir tamaño de la imagen
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


# Copia primero el archivo de dependencias
# Esto permite aprovechar el cache de Docker si requirements.txt no cambia
COPY requirements.txt .

# Actualiza pip
# Instala dependencias del proyecto
# Instala también explícitamente fastapi, uvicorn y otras librerías necesarias
# --no-cache-dir evita guardar cache y reduce tamaño final
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir fastapi "uvicorn[standard]" jinja2 joblib


# Copia todo el código del proyecto al contenedor
COPY . .

# Expone el puerto 8001 (documentación, no abre firewall)
# Indica qué puerto usará la aplicación
EXPOSE 8001

# Comando que se ejecuta cuando el contenedor inicia
# Ejecuta Uvicorn apuntando al objeto app dentro de src/api/app.py
# --host 0.0.0.0 permite que sea accesible desde fuera del contenedor
# --port 8001 define el puerto interno
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8001"]