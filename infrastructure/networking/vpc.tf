resource "google_compute_network" "vpc_network" {
  name                    = "networksecurity-vpc"
  auto_create_subnetworks = true
}
