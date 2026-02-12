# ------------------------------------------------------
# Imagen base
# ------------------------------------------------------
# Usamos la imagen oficial LTS de Jenkins.
# Esta imagen ya trae Jenkins instalado y configurado.
# Actualmente está basada en Debian moderno (trixie).
FROM jenkins/jenkins:lts


# ------------------------------------------------------
# Cambiamos a usuario root
# ------------------------------------------------------
# Jenkins corre por defecto como usuario "jenkins".
# Para instalar paquetes del sistema necesitamos permisos de root.
USER root


# ------------------------------------------------------
# Instalación de dependencias básicas del sistema
# ------------------------------------------------------
# Instalamos:
# - ca-certificates → certificados SSL
# - curl            → descargar archivos desde internet
# - gnupg           → manejar claves GPG
RUN apt-get update -y && \
    apt-get install -y \
        ca-certificates \
        curl \
        gnupg && \
    apt-get clean


# ------------------------------------------------------
# Añadir la clave GPG oficial de Docker (método moderno)
# ------------------------------------------------------
# Debian moderno ya NO soporta "apt-key".
# Usamos /etc/apt/keyrings como recomienda Docker.
RUN mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/debian/gpg | \
    gpg --dearmor -o /etc/apt/keyrings/docker.gpg


# ------------------------------------------------------
# Añadir el repositorio oficial de Docker
# ------------------------------------------------------
# Aunque Jenkins use Debian trixie,
# Docker publica paquetes estables para bullseye,
# que son totalmente compatibles.
RUN echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/debian bullseye stable" \
> /etc/apt/sources.list.d/docker.list


# ------------------------------------------------------
# Instalación de Docker dentro del contenedor
# ------------------------------------------------------
# Instalamos:
# - docker-ce        → Docker Engine
# - docker-ce-cli    → Cliente Docker
# - containerd.io    → Runtime de contenedores
RUN apt-get update -y && \
    apt-get install -y \
        docker-ce \
        docker-ce-cli \
        containerd.io && \
    apt-get clean


# ------------------------------------------------------
# Configuración de permisos para Jenkins
# ------------------------------------------------------
# Creamos el grupo "docker" si no existe.
# Añadimos el usuario "jenkins" a ese grupo.
# Esto permite ejecutar comandos docker sin sudo.
RUN groupadd -f docker && \
    usermod -aG docker jenkins


# ------------------------------------------------------
# Preparación del directorio de Docker (DinD)
# ------------------------------------------------------
# Directorio donde Docker guarda imágenes y contenedores.
# Necesario cuando se usa Docker dentro de Docker.
RUN mkdir -p /var/lib/docker


# ------------------------------------------------------
# Volumen persistente para Docker
# ------------------------------------------------------
# Permite que las imágenes y capas de Docker
# sobrevivan a reinicios del contenedor.
VOLUME /var/lib/docker


# ------------------------------------------------------
# Volvemos al usuario Jenkins
# ------------------------------------------------------
# Buena práctica de seguridad:
# Jenkins y los pipelines NO deben correr como root.
USER jenkins
