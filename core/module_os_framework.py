class ModuleOS:
    def __init__(self, module_name, parent_fabric=None):
        self.module_name = module_name
        self.running = True
        self.parent_fabric = parent_fabric
        self.intelligence_layer = None
        self.persistence_handler = None
        self.network_node = None
        
    async def boot_module_os(self):
        """Boot sequence for specialized module OS"""
        logger.info(f"🚀 Booting {self.module_name} OS")
        
        try:
            # 1. Initialize module-specific intelligence
            await self._initialize_intelligence()
            
            # 2. Connect to persistence layer
            await self._connect_persistence()
            
            # 3. Establish network connectivity
            await self._establish_networking()
            
            # 4. Register with main fabric
            await self._register_with_fabric()
            
            logger.info(f"✅ {self.module_name} OS fully operational")
            return self
            
        except Exception as e:
            logger.error(f"💥 {self.module_name} OS boot failed: {e}")
            raise

    async def _initialize_intelligence(self):
        """Initialize 14B parameter intelligence for this module"""
        from module_intelligence import create_module_brain
        self.intelligence_layer = await create_module_brain(
            module_type=self.module_name,
            fabric_context=self.parent_fabric
        )
        
    async def _connect_persistence(self):
        """Yjs-based persistence for module state"""
        from yjs_persistence import create_module_persistence
        self.persistence_handler = await create_module_persistence(
            module_id=self.module_name,
            fabric=self.parent_fabric
        )
        
    async def _establish_networking(self):
        """AnyNode networking for module communication"""
        from anynode_network import create_module_node
        self.network_node = await create_module_node(
            module_name=self.module_name,
            fabric=self.parent_fabric.complete_fabric
        )
        
    async def _register_with_fabric(self):
        """Register this module OS with the main Nexus fabric"""
        if self.parent_fabric:
            await self.parent_fabric.register_module(
                module_name=self.module_name,
                module_os=self,
                capabilities=self.get_capabilities()
            )