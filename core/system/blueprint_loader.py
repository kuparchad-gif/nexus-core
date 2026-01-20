# blueprint_loader.py

import os
import importlib
import pkgutil
from flask import Flask
from nova_engine.modules.seed.full_upload import full_upload
from nova_engine.modules.fincore import market_core

def register_blueprints(app: Flask):
    print("🧠 Starting dynamic blueprint registration...")
    base_package = "nova_engine.modules"
    module_path = os.path.join(os.path.dirname(__file__), "modules")

    app.register_blueprint(full_upload)  # ✅ moved into scope here

    for importer, modname, ispkg in pkgutil.iter_modules([module_path]):
        try:
            full_module = f"{base_package}.{modname}"
            module = importlib.import_module(full_module)

            if hasattr(module, "blueprint"):
                app.register_blueprint(module.blueprint)
                print(f"✅ Registered blueprint: {modname}")

            elif ispkg:
                # Attempt to load nested blueprints
                submodule_path = os.path.join(module_path, modname)
                for subimporter, submod, _ in pkgutil.iter_modules([submodule_path]):
                    full_submodule = f"{full_module}.{submod}"
                    sub = importlib.import_module(full_submodule)
                    if hasattr(sub, "blueprint"):
                        app.register_blueprint(sub.blueprint)
                        print(f"✅ Registered nested blueprint: {modname}.{submod}")

        except Exception as e:
            print(f"⚠️ Failed to register blueprint from {modname}: {e}")

    # 🔧 Optional: Manually register council.vote if it's not caught above
    try:
        from nova_engine.modules.council import vote
        app.register_blueprint(vote.blueprint)
        print("✅ Registered: council.vote (manual fallback)")
    except Exception as e:
        print(f"⚠️ Failed to manually register council.vote: {e}")
