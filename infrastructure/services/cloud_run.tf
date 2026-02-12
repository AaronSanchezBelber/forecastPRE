resource "google_cloud_run_service" "app" {
  name     = "networksecurity"
  location = var.region

  template {
    spec {
      containers {
        image = "gcr.io/${var.project_id}/networksecurity:latest"
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }
}

output "cloud_run_url" {
  value = google_cloud_run_service.app.status[0].url
}
