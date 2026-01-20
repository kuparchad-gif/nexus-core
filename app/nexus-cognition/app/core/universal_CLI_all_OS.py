# universal_cli.py
class UniversalCLI:
    """One CLI to rule all OS instances - Heroku-style commands for everything"""
    
    def __init__(self):
        self.os_registry = {}  # name -> OS instance
        self.command_registry = self._build_command_registry()
        
    def register_os(self, name: str, os_instance):
        """Register an OS instance with the CLI"""
        self.os_registry[name] = os_instance
        print(f"✅ Registered: {name}")
    
    async def handle_command(self, command: str, args: List[str]) -> Dict:
        """Route commands to appropriate OS instances"""
        if command == "ps":
            return await self._show_all_status()
        elif command == "health":
            return await self._system_health_check()
        elif command == "restart":
            return await self._restart_os(args[0] if args else "all")
        elif command in self.os_registry:
            return await self._route_to_os(command, args)
        else:
            return {"error": f"Unknown command: {command}"}
    
    async def _show_all_status(self) -> Dict:
        """Show status of all OS instances (Heroku 'ps' style)"""
        statuses = {}
        for name, os_instance in self.os_registry.items():
            try:
                if hasattr(os_instance, 'get_system_health'):
                    statuses[name] = await os_instance.get_system_health()
                else:
                    statuses[name] = {"status": "active", "details": "No health method"}
            except Exception as e:
                statuses[name] = {"status": "error", "error": str(e)}
        
        return {"system_status": statuses}