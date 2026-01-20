# Path: nexus_platform/common/threed_world.py
import panda3d.core as p3d
from direct.showbase.ShowBase import ShowBase

class ThreeDWorld:
    def __init__(self, service_name: str):
        self.base = ShowBase()
        self.logger = setup_logger(f"{service_name}.3dworld")

    def initialize_world(self, model_path: str):
        try:
            model = self.base.loader.loadModel(model_path)
            model.reparentTo(self.base.render)
            self.logger.info({"action": "3d_world_initialized", "model": model_path})
        except Exception as e:
            self.logger.error({"action": "3d_world_init_failed", "error": str(e)})
            raise

    def render_frame(self):
        self.base.taskMgr.step()