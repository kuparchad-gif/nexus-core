import json
import os

class MissionAssignmentAPI:
    def __init__(self, drone_command_sheet_path, mission_log_path='/Utilities/manifest/mission_log.json'):
        self.command_sheet_path = drone_command_sheet_path
        self.mission_log_path = mission_log_path
        self.drones = self.load_drones()

    def load_drones(self):
        if not os.path.exists(self.command_sheet_path):
            raise FileNotFoundError(f"Drone Command Sheet not found at {self.command_sheet_path}")
        with open(self.command_sheet_path, 'r') as file:
            data = json.load(file)
        return data.get("drone_fleet", {})

    def assign_mission(self, drone_id, mission_description):
        if drone_id not in self.drones:
            return f"⚠️ Drone {drone_id} not found."

        mission_entry = {
            "drone_id": drone_id,
            "mission": mission_description,
            "status": "Assigned"
        }

        if not os.path.exists(self.mission_log_path):
            mission_log = []
        else:
            with open(self.mission_log_path, 'r') as file:
                mission_log = json.load(file)

        mission_log.append(mission_entry)

        with open(self.mission_log_path, 'w') as file:
            json.dump(mission_log, file, indent=4)

        return f"🚀 Mission assigned to {drone_id}: {mission_description}"

    def list_missions(self):
        if not os.path.exists(self.mission_log_path):
            return []
        with open(self.mission_log_path, 'r') as file:
            return json.load(file)

# Example Usage:
# mission_api = MissionAssignmentAPI('/Utilities/drone_core/drone_command_sheet.json')
# mission_api.assign_mission('Explorer-01', 'Map the Outer Net Realms.')
# mission_api.list_missions()
