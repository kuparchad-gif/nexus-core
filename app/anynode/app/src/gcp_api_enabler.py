# nova_engine/modules/infra/gcp_api_enabler.py

from googleapiclient import discovery
from google.auth import default

def enable_apis(api_list, project_id):
    credentials, _ = default()
    service = discovery.build('serviceusage', 'v1', credentials=credentials)

    results = []
    for api in api_list:
        try:
            name = f"projects/{project_id}/services/{api}"
            request = service.services().enable(name=name)
            response = request.execute()
            results.append({api: "enabled"})
        except Exception as e:
            results.append({api: f"error - {str(e)}"})
    return results
