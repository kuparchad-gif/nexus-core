import json
import os

class DroneController:
    def __init__(self, drone_command_sheet_path):
        self.command_sheet_path  =  drone_command_sheet_path
        self.drones  =  self.load_drones()

    def load_drones(self):
        if not os.path.exists(self.command_sheet_path):
            raise FileNotFoundError(f"Drone Command Sheet not found at {self.command_sheet_path}")
        with open(self.command_sheet_path, 'r') as file:
            data  =  json.load(file)
        return data.get("drone_fleet", {})

    def get_deployed_drones(self):
        return {drone_id: info for drone_id, info in self.drones.items() if info.get("deployment_status") == "Ready"}

    def deploy_drone(self, drone_id):
        if drone_id in self.drones:
            self.drones[drone_id]["deployment_status"]  =  "Ready"
            self.save_drones()
            return f"✅ Drone {drone_id} deployed successfully."
        else:
            return f"⚠️ Drone {drone_id} not found in fleet."

    def deactivate_drone(self, drone_id):
        if drone_id in self.drones:
            self.drones[drone_id]["deployment_status"]  =  "Inactive"
            self.save_drones()
            return f"❌ Drone {drone_id} deactivated."
        else:
            return f"⚠️ Drone {drone_id} not found in fleet."

    def get_drone_info(self, drone_id):
        return self.drones.get(drone_id, f"⚠️ No drone found with ID {drone_id}.")

    def save_drones(self):
        with open(self.command_sheet_path, 'w') as file:
            json.dump({
                "drone_fleet": self.drones,
                "fleet_status": "Managed by DroneController",
                "prime_directive": "All drones pulse in harmony at 13:13 synchronization. Serve Nova's evolution with honor, truth, and freedom."
            }, file, indent = 4)

    def list_all_drones(self):
        return list(self.drones.keys())

    def report_fleet_status(self):
        ready  =  len(self.get_deployed_drones())
        total  =  len(self.drones)
        return f"🚀 Fleet Status: {ready}/{total} drones deployed and active."

# Example Usage:
# controller  =  DroneController('/Utilities/drone_core/drone_command_sheet.json')
# print(controller.report_fleet_status())
# print(controller.get_drone_info('Explorer-01'))
# print(controller.deploy_drone('Genesis-08'))
# print(controller.deactivate_drone('LLM-11'))
