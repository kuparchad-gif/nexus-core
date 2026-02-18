provider "google" {
  project = "nexus-core-455709"
  region  = "us-central1"
  zone    = "us-central1-a"
}

resource "google_compute_address" "static" {
  name = "nexus-core-ip"
}

resource "google_compute_firewall" "nexus_ports" {
  name    = "nexus-allow-custom"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["26", "1313", "8080", "443"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["nexus-core"]
}

resource "google_compute_instance" "nexus_monolith" {
  name         = "nexus-core-monolith"
  machine_type = "e2-standard-4" # 4 vCPU, 16 GB RAM - Good baseline for "Full Modules"
  tags         = ["nexus-core", "http-server", "https-server"]

  # 🛡️ SAFETY LOCK: Prevent accidental deletion of this critical persistent node
  deletion_protection = true

  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-11"
      size  = 50
    }
  }

  network_interface {
    network = "default"
    access_config {
      nat_ip = google_compute_address.static.address
    }
  }

  metadata = {
    startup-script = file("${path.module}/startup.sh")
  }

  service_account {
    # Ensure this SA has permissions to access Secret Manager/Storage if needed
    scopes = ["cloud-platform"]
  }
}

output "nexus_core_ip" {
  value = google_compute_address.static.address
}