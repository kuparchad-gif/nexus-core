# Systems/engine/nova_drone.py

class Drone:
    def __init__(self, diagnostics, task_log):
        self.diagnostics = diagnostics
        self.task_log = task_log
        self.role = None
        self.next_task = None

    def vote_for_role(self):
        # Simple role logic — more can be added
        if self.diagnostics['api_load'] > 0.7:
            self.role = 'Orc'
        elif self.task_log and 'schedule' in self.task_log[-1]:
            self.role = 'Sched'
        elif 'voice' in self.task_log[-1]:
            self.role = 'Tone'
        elif 'message' in self.task_log[-1]:
            self.role = 'Txt'
        else:
            self.role = 'Mem'
        return self.role

    def perform_role(self):
        print(f"Performing role: {self.role}")
        # Load corresponding behavior modules
        if self.role == 'Orc':
            from systems.engine.api import orchestrator_api
            orchestrator_api.run()
        elif self.role == 'Sched':
            from systems.engine.process import scheduler
            scheduler.run()
        elif self.role == 'Tone':
            from systems.engine.modules import tone_analyzer
            tone_analyzer.analyze()
        elif self.role == 'Txt':
            from systems.engine.modules import text_handler
            text_handler.handle()
        elif self.role == 'Mem':
            from systems.engine.memory import stream_logger
            stream_logger.write_logs()
