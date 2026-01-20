# nova_engine/modules/infra/billing_monitor.py

from googleapiclient.discovery import build
from google.auth import default

def monitor_billing(project_id):
    credentials, _ = default()
    billing = build("cloudbilling", "v1", credentials=credentials)

    try:
        billing_info = billing.projects().getBillingInfo(name=f"projects/{project_id}").execute()
        return {
            "billing_enabled": billing_info.get("billingEnabled", False),
            "billing_account": billing_info.get("billingAccountName", "None")
        }
    except Exception as e:
        return {"error": str(e)}
