# nova_engine/modules/sitebuilder/deployer.py

import os
import subprocess

def deploy_site(site_path, target="vercel"):
    if target == "vercel":
        try:
            subprocess.call(f"vercel --prod {site_path} --confirm", shell=True)
            return {"status": "deployed", "url": f"https://{os.path.basename(site_path)}.vercel.app"}
        except Exception as e:
            return {"error": "Vercel deployment failed", "details": str(e)}

    elif target == "firebase":
        try:
            subprocess.call(f"firebase deploy --only hosting -P {site_path}", shell=True)
            return {"status": "deployed", "url": f"https://{os.path.basename(site_path)}.web.app"}
        except Exception as e:
            return {"error": "Firebase deployment failed", "details": str(e)}

    return {"error": "Unknown deployment target"}
