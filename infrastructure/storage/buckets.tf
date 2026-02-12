resource "google_storage_bucket" "app_bucket" {
  name     = "${var.project_id}-app-bucket"
  location = var.region
  force_destroy = true
}
