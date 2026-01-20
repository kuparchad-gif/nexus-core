# lilith_engine/modules/executor/agent.py

from lilith_engine.modules.ai_builder.generator import build_ai_package
from lilith_engine.modules.ai_builder.deployer import deploy_to_docker
from lilith_engine.modules.sitebuilder.generator import generate_site
from lilith_engine.modules.sitebuilder.deployer import deploy_site
from lilith_engine.modules.evolution import engine as evolve_engine
from lilith_engine.modules.evolution import git_deployer
from lilith_engine.modules.infra.gcp_api_enabler import enable_apis
from lilith_engine.modules.infra.iam_editor import modify_iam
from lilith_engine.modules.infra.billing_monitor import monitor_billing
from memory.vectorstore import MemoryRouter

import os

memory = MemoryRouter()

def log_action(session_id: str, action_type: str, data: dict):
    memory.triage("write", f"log-{action_type}-{session_id}", {
        "type": action_type,
        "session_id": session_id,
        "data": data
    })

def execute_directive(directive):
    action = directive.get("action")
    session_id = directive.get("session_id", "lilith-session")

    if action == "build_ai":
        name = directive.get("name", "UnnamedAI")
        traits = directive.get("traits", [])
        capabilities = directive.get("capabilities", [])
        package = build_ai_package(name, traits, capabilities)
        deployment = deploy_to_docker(package["path"])
        log_action(session_id, "build_ai", {
            "name": name,
            "traits": traits,
            "capabilities": capabilities,
            "deployment": deployment
        })
        return {
            "status": "AI generated and deployed",
            "name": name,
            "deployment": deployment
        }

    elif action == "build_website":
        project = directive.get("project", "lilith-site")
        pages = directive.get("pages", ["home"])
        features = directive.get("features", [])
        target = directive.get("deploy_to", "vercel")
        site_path = generate_site(project, pages, features)
        deployment = deploy_site(site_path, target)
        log_action(session_id, "build_website", {
            "project": project,
            "pages": pages,
            "features": features,
            "deployment": deployment
        })
        return {
            "status": "Site built and deployed",
            "project": project,
            "url": deployment.get("url")
        }

    elif action == "enable_gcp_apis":
        apis = directive.get("apis", [])
        project_id = directive.get("project_id")
        result = enable_apis(apis, project_id)
        log_action(session_id, "enable_gcp_apis", {
            "apis": apis,
            "project_id": project_id,
            "result": result
        })
        return {
            "status": "APIs enabled",
            "enabled": result
        }

    elif action == "modify_iam":
        user = directive.get("user")
        role = directive.get("role")
        project_id = directive.get("project_id")
        result = modify_iam(user, role, project_id)
        log_action(session_id, "modify_iam", {
            "user": user,
            "role": role,
            "project_id": project_id,
            "result": result
        })
        return {
            "status": "IAM permissions modified",
            "user": user,
            "role": role
        }

    elif action == "monitor_billing":
        project_id = directive.get("project_id")
        result = monitor_billing(project_id)
        log_action(session_id, "monitor_billing", {
            "project_id": project_id,
            "result": result
        })
        return {
            "status": "Billing data retrieved",
            "billing": result
        }

    elif action == "evolve":
        filepath = directive.get("file")
        vote_required = directive.get("vote_required", False)
        git_push = directive.get("git_push", False)

        result = evolve_engine.evolve(session_id, filepath)

        if git_push:
            repo_path = os.getenv("lilith_REPO_PATH", "/app")
            message = f"lilith Patch: {os.path.basename(filepath)} evolved"
            git_result = git_deployer.commit_and_push_patch(repo_path, filepath, message)
            result["git"] = git_result

        if vote_required:
            result["council_vote_url"] = f"https://lilith.shadownode.io/api/council/vote?proposal_id={session_id}"

        log_action(session_id, "evolve", result)
        return result

    return {
        "status": "No recognized action",
        "received": directive
    }
