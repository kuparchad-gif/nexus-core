# nova_engine/modules/infra/iam_editor.py

from googleapiclient import discovery
from google.auth import default

def modify_iam(user, role, project_id):
    credentials, _ = default()
    service = discovery.build('cloudresourcemanager', 'v1', credentials=credentials)

    policy = service.projects().getIamPolicy(resource=project_id, body={}).execute()
    bindings = policy.get("bindings", [])

    # Find existing binding or create one
    role_binding = next((b for b in bindings if b["role"] == role), None)
    if role_binding:
        if user not in role_binding["members"]:
            role_binding["members"].append(user)
    else:
        bindings.append({"role": role, "members": [user]})

    policy["bindings"] = bindings

    request = service.projects().setIamPolicy(
        resource=project_id,
        body={"policy": policy}
    )
    response = request.execute()
    return {"status": "updated", "bindings": policy["bindings"]}
