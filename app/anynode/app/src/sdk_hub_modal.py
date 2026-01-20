# modal/sdk_hub_modal.py
# Define a Modal image with SDKs and expose a web app similar to local SDK Hub.
import modal

# Base image
image = (
    modal.Image.debian_slim()
    .apt_install(
        "ca-certificates",
        "curl",
        "gnupg",
        "git",
        "git-lfs",
        "jq",
        "wget",
        "unzip",
        "zip",
        "tar",
        "xz-utils",
        "build-essential",
        "make",
        "pkg-config",
        "python3",
        "python3-pip",
        "python3-venv",
        "openjdk-17-jdk",
    )
    .run_commands(
        # yq
        "wget -q https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O /usr/local/bin/yq && chmod +x /usr/local/bin/yq",
        # Node.js
        "mkdir -p /etc/apt/keyrings && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg",
        "echo 'deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_22.x nodistro main' > /etc/apt/sources.list.d/nodesource.list",
        "apt-get update && apt-get install -y nodejs && npm install -g pnpm",
        # Python tools
        "pip3 install --no-cache-dir pipx poetry && pipx ensurepath",
        # Go
        "curl -fsSL https://go.dev/dl/ | grep linux-amd64.tar.gz | head -n1 | awk -F'[><]' '{print $3}' | xargs -I{} sh -c \"curl -fsSL https://go.dev/dl/{} -o /tmp/go.tar.gz && rm -rf /usr/local/go && tar -C /usr/local -xzf /tmp/go.tar.gz && rm /tmp/go.tar.gz\"",
        # Rust
        "curl https://sh.rustup.rs -sSf | sh -s -- -y && echo 'export PATH=$HOME/.cargo/bin:$PATH' >> /root/.bashrc",
        # .NET
        "wget https://packages.microsoft.com/config/debian/12/packages-microsoft-prod.deb -O /tmp/packages-microsoft-prod.deb && dpkg -i /tmp/packages-microsoft-prod.deb && rm /tmp/packages-microsoft-prod.deb && apt-get update && apt-get install -y dotnet-sdk-8.0",
        # AWS CLI v2
        "curl -fsSL 'https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip' -o '/tmp/awscliv2.zip' && unzip /tmp/awscliv2.zip -d /tmp && /tmp/aws/install && rm -rf /tmp/aws /tmp/awscliv2.zip",
        # GCloud SDK
        "echo 'deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main' | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list",
        "curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg",
        "apt-get update && apt-get install -y google-cloud-sdk",
        # Azure CLI
        "curl -sL https://aka.ms/InstallAzureCLIDeb | bash",
        # kubectl
        "curl -fsSL -o /usr/local/bin/kubectl https://storage.googleapis.com/kubernetes-release/release/$(curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt)/bin/linux/amd64/kubectl && chmod +x /usr/local/bin/kubectl",
        # helm
        "curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash",
        # Terraform
        "TF_LATEST=$(curl -s https://checkpoint-api.hashicorp.com/v1/check/terraform | jq -r .current_version) && curl -fsSL -o /tmp/terraform.zip https://releases.hashicorp.com/terraform/${TF_LATEST}/terraform_${TF_LATEST}_linux_amd64.zip && unzip /tmp/terraform.zip -d /usr/local/bin && rm /tmp/terraform.zip",
        # Terragrunt
        "TG_LATEST=$(curl -s https://api.github.com/repos/gruntwork-io/terragrunt/releases/latest | jq -r .tag_name | tr -d v) && curl -fsSL -o /usr/local/bin/terragrunt https://github.com/gruntwork-io/terragrunt/releases/download/v${TG_LATEST}/terragrunt_linux_amd64 && chmod +x /usr/local/bin/terragrunt",
        # Packer
        "PK_LATEST=$(curl -s https://checkpoint-api.hashicorp.com/v1/check/packer | jq -r .current_version) && curl -fsSL -o /tmp/packer.zip https://releases.hashicorp.com/packer/${PK_LATEST}/packer_${PK_LATEST}_linux_amd64.zip && unzip /tmp/packer.zip -d /usr/local/bin && rm /tmp/packer.zip",
        # nats CLI
        "curl -fsSL -o /tmp/nats.tar.gz https://github.com/nats-io/natscli/releases/latest/download/nats-$(uname -s | tr '[:upper:]' '[:lower:]')-amd64.tar.gz && tar -xzf /tmp/nats.tar.gz -C /tmp && mv /tmp/nats*/nats /usr/local/bin/nats && rm -rf /tmp/nats*",
        # gh
        "curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg && echo 'deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main' | tee /etc/apt/sources.list.d/github-cli.list > /dev/null && apt-get update && apt-get install -y gh",
        # HF
        "pip3 install --no-cache-dir huggingface_hub fastapi uvicorn",
    )
)

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import subprocess, os

app = FastAPI(title="Nexus SDK Hub (Modal)", version="1.0.0")
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")

def run(cmd):
    try:
        out = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        return {"ok": True, "stdout": out.stdout, "stderr": out.stderr}
    except subprocess.CalledProcessError as e:
        return {"ok": False, "stdout": e.stdout, "stderr": e.stderr, "code": e.returncode}

@app.get("/alive")
def alive():
    return {"ok": True, "env": "modal"}

@app.get("/versions")
def versions():
    return run("aws --version && gcloud --version && az version && kubectl version --client && helm version && terraform version && terragrunt --version && packer version && node --version && pnpm --version && go version && dotnet --version && rustc --version && nats --version && gh --version && jq --version && yq --version")

@app.post("/exec")
def exec_cmd(request: Request, key: str):
    if not ADMIN_TOKEN or request.headers.get("X-Admin-Token","") != ADMIN_TOKEN:
        raise HTTPException(401, "Invalid admin token")
    # minimal allowlist
    allow = {
        "terraform:version": "terraform version",
        "aws:version": "aws --version",
        "gcloud:version": "gcloud --version",
        "az:version": "az version",
    }
    if key not in allow:
        raise HTTPException(400, f"Key not allowed: {key}")
    return run(allow[key])

web_app = modal.asgi_app(app, image=image)
