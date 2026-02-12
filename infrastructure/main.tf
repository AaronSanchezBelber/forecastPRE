# Llamado a módulos o recursos principales
module "networking" {
  source = "./networking"
}

module "services" {
  source = "./services"
}

module "storage" {
  source = "./storage"
}
