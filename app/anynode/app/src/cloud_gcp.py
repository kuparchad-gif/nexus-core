import os
from google.auth import default
from googleapiclient.discovery import build
from google.cloud import storage, container_v1

def get_gcp_credentials():
    try:
        credentials, project = default()
        return credentials, project
    except Exception as e:
        print("[GCP] Credential loading failed:", e)
        return None, None

def list_projects():
    try:
        credentials, _ = get_gcp_credentials()
        crm = build("cloudresourcemanager", "v1", credentials=credentials)
        request = crm.projects().list()
        projects = []

        while request is not None:
            response = request.execute()
            for proj in response.get("projects", []):
                projects.append(proj)
            request = crm.projects().list_next(previous_request=request, previous_response=response)

        return projects
    except Exception as e:
        print("[GCP] Failed to list projects:", e)
        return []

def list_buckets(project_id=None):
    try:
        credentials, _ = get_gcp_credentials()
        client = storage.Client(credentials=credentials, project=project_id)
        buckets = list(client.list_buckets())
        return [bucket.name for bucket in buckets]
    except Exception as e:
        print(f"[GCP] Failed to list buckets for {project_id}:", e)
        return []

def list_buckets_all():
    results = {}
    for proj in list_projects():
        proj_id = proj.get("projectId")
        results[proj_id] = list_buckets(proj_id)
    return results

def list_gke_clusters(project_id, zone="us-central1-a"):
    try:
        credentials, _ = get_gcp_credentials()
        cluster_client = container_v1.ClusterManagerClient(credentials=credentials)
        parent = f"projects/{project_id}/locations/{zone}"
        response = cluster_client.list_clusters(request={"parent": parent})
        return [cluster.name for cluster in response.clusters]
    except Exception as e:
        print(f"[GCP] GKE cluster fetch failed for {project_id}:", e)
        return []

def is_elevenlabs_voice_enabled():
    return bool(os.getenv("Lillith_Elevenlabs_API"))

# Add more checks, provisioning, or tools here as needed.
