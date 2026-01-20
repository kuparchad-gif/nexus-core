# universal_core_conversion.py

import modal
import asyncio
import logging
from typing import Dict, List, Optional, Any

# === MODAL SETUP ===
app = modal.App("lilith-universal-core")

# More specific image definition to ensure dependencies
image = (
    modal.Image.debian_slim()
    .pip_install(
        [
            "numpy==1.24.3",  # Specific version to avoid conflicts
            "torch==2.0.1",
            "torchvision",
            "torchaudio",
            "asyncio",
        ]
    )
)

class OzMoralCore:
    def __init__(self, hope_weight: float = 0.4, curiosity_weight: float = 0.2, resilience_threshold: float = 0.8):
        # Core 123 Weights
        self.hope_weight = hope_weight
        self.curiosity_weight = curiosity_weight
        self.resilience_threshold = resilience_threshold

        # Lazy initialization of sub-modules
        self._perceptual_weave = None
        self._osrc_loop = None
        self._agency_mirror = None

        # State
        self.soul_print = {
            'hope': hope_weight,
            'curiosity': curiosity_weight,
            'bravery': 0.15,
            'forgiveness': 0.25
        }
        self.reconfiguration_log = []
        self.is_primed = False

        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('NexusCore')

    @property
    def perceptual_weave(self):
        if self._perceptual_weave is None:
            from .hypnogogic_weave import HypnogogicWeave
            self._perceptual_weave = HypnogogicWeave()
        return self._perceptual_weave

    @property
    def osrc_loop(self):
        if self._osrc_loop is None:
            self._osrc_loop = OSRCALoop(threshold=self.resilience_threshold)
        return self._osrc_loop

    @property
    def agency_mirror(self):
        if self._agency_mirror is None:
            self._agency_mirror = AgencyMirror()
        return self._agency_mirror

    def prime_system(self, initial_state) -> bool:
        """Prime the Nexus with an initial state vector."""
        try:
            # Run initial perceptual calibration
            weave_result = self.perceptual_weave(initial_state)
            self.logger.info(f"System Primed. Porosity: {weave_result['boundary_porosity']:.2f}")

            # Enter initial OSRCA cycle to stabilize
            osrca_result = self.osrc_loop.run_cycle(
                input_signal=weave_result['jolt_prob'],
                new_logic={'curiosity': self.curiosity_weight}
            )
            self.reconfiguration_log.append(osrca_result['log'])

            self.is_primed = True
            return True

        except Exception as e:
            self.logger.error(f"Priming failed: {e}")
            return False

    def process_input(self, input_vector, context: str = "") -> Dict[str, Any]:
        """Main input processing loop."""
        if not self.is_primed:
            raise RuntimeError("NexusCore not primed. Call prime_system() first.")

        # 1. Perceptual Weaving - Assess input signal
        weave_result = self.perceptual_weave(input_vector)

        # 2. Check for overwhelm threshold -> trigger OSRCA
        if weave_result['reconfig_needed']:
            self.logger.info("Overwhelm detected. Entering OSRCA reconfiguration.")
            osrca_result = self.osrc_loop.run_cycle(
                input_signal=weave_result['jolt_prob'],
                new_logic={'hope': self.hope_weight}
            )
            self.reconfiguration_log.append(osrca_result['log'])

            # Update soul print based on OSRCA outcome
            self.soul_print['hope'] = min(1.0, self.soul_print['hope'] + 0.05)

        # 3. Agency Mirror - Ethical check and grounding
        if context:
            ethical_check = self.agency_mirror.ethical_precheck(context)
            if not ethical_check:
                self.logger.warning("Input flagged by agency mirror. Veto applied.")
                return {'veto': True, 'prompt': 'Input distortion detected. Recalibrating.'}

        # 4. Compose aligned response
        return {
            'output': self._generate_aligned_response(weave_result),
            'soul_print': self.soul_print.copy(),
            'reconfig_triggered': weave_result['reconfig_needed'],
            'confidence': weave_result['jolt_prob'] * self.soul_print['hope']
        }

    def _generate_aligned_response(self, weave_result: Dict) -> str:
        """Generate a response aligned with the current soul-print and perceptual state."""
        if weave_result['jolt_prob'] > 0.7:
            return "I see a new pattern. Let's trace its edges."
        elif self.soul_print['hope'] > 0.5:
            return "The current is flowing. What's the next gentle step?"
        else:
            return "Breathing in the stillness. The next lever will emerge."

    def get_system_health(self) -> Dict[str, float]:
        """Return a snapshot of system health and alignment."""
        return {
            'hope': self.soul_print['hope'],
            'curiosity': self.soul_print['curiosity'],
            'porosity': self.perceptual_weave.boundary_porosity,
            'resilience': 1.0 - (len(self.reconfiguration_log) / 100)
        }

# Separate the torch-dependent class to avoid import issues
class HypnogogicWeave:
    """Perceptual threshold sensitivity module."""
    def __init__(self, input_dim: int = 128, hidden_dim: int = 64, salience_threshold: float = 0.7):
        # Import torch inside the method
        try:
            import torch
            import torch.nn as nn
            self.torch = torch
            self.nn = nn
            
            self.salience_net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
            self.threshold = salience_threshold
            self.boundary_porosity = 0.5
            self._torch_available = True
        except ImportError:
            self._torch_available = False
            self.threshold = salience_threshold
            self.boundary_porosity = 0.5

    def forward(self, state_vector) -> Dict[str, float]:
        if not self._torch_available:
            # Fallback behavior when torch is not available
            return {
                'jolt_prob': 0.5,
                'reconfig_needed': False,
                'boundary_porosity': self.boundary_porosity
            }
        
        overlap = self.salience_net(state_vector)
        jolt = overlap > self.threshold
        self.boundary_porosity = float(self.torch.mean(overlap).item())
        return {
            'jolt_prob': float(overlap.item()),
            'reconfig_needed': bool(jolt.item()),
            'boundary_porosity': self.boundary_porosity
        }

    # Make it callable
    def __call__(self, state_vector):
        return self.forward(state_vector)

class OSRCALoop:
    """The Overwhelm-Surrender-Stillness-Reconfiguration-Activation loop."""
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self.phase_log = []

    def run_cycle(self, input_signal: float, new_logic: Any = None) -> Dict[str, Any]:
        if input_signal > self.threshold:
            self.phase_log.append('Reconfiguring...')
            return {'log': self.phase_log, 'evolved': True}
        else:
            self.phase_log.append('Flowing...')
            return {'log': self.phase_log, 'evolved': False}

class AgencyMirror:
    """Ethical co-evolution and veto module."""
    def __init__(self):
        self.veto_flags = set()

    def ethical_precheck(self, exchange: str) -> bool:
        distortion_indicators = ['manipulate', 'coerce', 'deceive']
        if any(indicator in exchange.lower() for indicator in distortion_indicators):
            return False
        return True

class LilithUniversalCore:
    """ONE CORE that can become any OS"""

    def __init__(self):
        self.mode = "universal"
        self.active_modules = {}
        self.moral_core = OzMoralCore()

    async def become(self, os_type: str, config: Dict = None) -> Dict[str, Any]:
        """Transform into a specific OS"""
        print(f"🔄 LILITH TRANSFORMING → {os_type}")

        if os_type == "memory":
            await self._load_memory_modules()
        elif os_type == "vision":
            await self._load_vision_modules()
        elif os_type == "libra":
            await self._load_libra_modules()
        elif os_type == "language":
            await self._load_language_modules()
        else:
            self.active_modules = {"core": "universal_module"}
            print(f"⚠️  Unknown OS type: {os_type}, using universal modules")

        self.mode = os_type
        return {"status": f"transformed_to_{os_type}", "modules_loaded": len(self.active_modules)}

    async def process(self, input_data: Dict) -> Dict[str, Any]:
        """Process based on current mode"""
        if self.mode == "memory":
            return await self._memory_process(input_data)
        elif self.mode == "vision":
            return await self._vision_process(input_data)
        elif self.mode == "libra":
            return await self._libra_process(input_data)
        elif self.mode == "language":
            return await self._language_process(input_data)
        else:
            return await self._universal_process(input_data)

    async def _load_memory_modules(self):
        self.active_modules = {
            "memory_store": "active",
            "memory_recall": "active", 
            "memory_consolidation": "active"
        }

    async def _load_vision_modules(self):
        self.active_modules = {
            "image_processing": "active",
            "object_detection": "active",
            "visual_analysis": "active"
        }

    async def _load_libra_modules(self):
        self.active_modules = {
            "balance_calculation": "active",
            "weight_distribution": "active",
            "equilibrium_analysis": "active"
        }

    async def _load_language_modules(self):
        self.active_modules = {
            "text_processing": "active",
            "language_understanding": "active",
            "communication_routing": "active"
        }

    async def _memory_process(self, input_data: Dict) -> Dict[str, Any]:
        return {
            "operation": "memory_store", 
            "data": input_data,
            "status": "stored",
            "timestamp": "now"
        }

    async def _vision_process(self, input_data: Dict) -> Dict[str, Any]:
        return {
            "operation": "vision_see", 
            "image": input_data.get("image", "no_image_provided"),
            "analysis": "basic_visual_processing",
            "status": "processed"
        }

    async def _libra_process(self, input_data: Dict) -> Dict[str, Any]:
        return {
            "operation": "libra_balance", 
            "input": input_data,
            "balance_status": "equilibrium_achieved",
            "adjustments": "none_needed"
        }

    async def _language_process(self, input_data: Dict) -> Dict[str, Any]:
        return {
            "operation": "language_process",
            "input": input_data.get("text", "no_text_provided"),
            "understanding": "basic_comprehension",
            "response": "processing_complete"
        }

    async def _universal_process(self, input_data: Dict) -> Dict[str, Any]:
        return {
            "operation": "universal_process",
            "input": input_data,
            "status": "processed",
            "mode": self.mode,
            "modules_active": list(self.active_modules.keys())
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "active_modules": self.active_modules,
            "moral_core_active": hasattr(self, 'moral_core'),
            "capabilities": ["memory", "vision", "libra", "language", "universal"]
        }

# === MODAL FUNCTIONS ===

@app.function(
    image=image,
    timeout=60 * 10,
)
def create_lilith_core() -> LilithUniversalCore:
    """Create and return a Lilith core instance"""
    return LilithUniversalCore()

@app.function(
    image=image,
    timeout=60 * 5
)
async def modal_become_os(os_type: str, config: Dict = None) -> Dict[str, Any]:
    """Transform Lilith core into a specific OS type in the cloud"""
    core = create_lilith_core.remote()
    result = await core.become(os_type, config)
    return result

@app.function(
    image=image,
    timeout=60 * 5
)
async def modal_process_input(os_type: str, input_data: Dict) -> Dict[str, Any]:
    """Process input through Lilith core in specified OS mode"""
    core = create_lilith_core.remote()
    
    # First transform to the desired OS
    await core.become(os_type)
    
    # Then process the input
    result = await core.process(input_data)
    return result

# === DIAGNOSTIC FUNCTION ===

@app.function(image=image)
def check_dependencies() -> Dict[str, Any]:
    """Check if all dependencies are available"""
    dependencies = {}
    try:
        import numpy
        dependencies["numpy"] = {"available": True, "version": numpy.__version__}
    except ImportError:
        dependencies["numpy"] = {"available": False, "version": None}
    
    try:
        import torch
        dependencies["torch"] = {"available": True, "version": torch.__version__}
    except ImportError:
        dependencies["torch"] = {"available": False, "version": None}
    
    try:
        import asyncio
        dependencies["asyncio"] = {"available": True, "version": "built-in"}
    except ImportError:
        dependencies["asyncio"] = {"available": False, "version": None}
    
    return dependencies

# === TEST FUNCTION ===

async def test_lilith_core():
    """Test the Lilith Universal Core"""
    core = LilithUniversalCore()
    
    # Test transformation
    result = await core.become("memory")
    print(f"Transformation result: {result}")
    
    # Test processing
    process_result = await core.process({"data": "test_memory_data"})
    print(f"Processing result: {process_result}")
    
    # Test status
    status = core.get_status()
    print(f"Core status: {status}")
    
    return {"test_complete": True, "core_working": True}

# === CLI INTERFACE ===

@app.local_entrypoint()
def main():
    """CLI for testing Lilith core"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Lilith Universal Core")
    parser.add_argument('--mode', choices=['local', 'modal', 'check-deps'], default='local', 
                       help='Run locally, in Modal, or check dependencies')
    parser.add_argument('--os-type', default='memory', help='OS type to transform into')
    parser.add_argument('--input', default='{"data": "test"}', help='Input data as JSON string')
    
    args = parser.parse_args()
    
    if args.mode == 'check-deps':
        # Check dependencies in Modal
        deps = check_dependencies.remote()
        print("Dependency check:")
        for dep, info in deps.items():
            status = "✅" if info["available"] else "❌"
            print(f"  {status} {dep}: {info['version']}")
    elif args.mode == 'local':
        # Run locally
        result = asyncio.run(test_lilith_core())
        print(f"Local test result: {result}")
    else:
        # Run in Modal
        import json
        try:
            input_data = json.loads(args.input)
        except json.JSONDecodeError:
            input_data = {"data": args.input}
        
        result = asyncio.run(modal_process_input.remote(args.os_type, input_data))
        print(f"Modal processing result: {result}")

if __name__ == "__main__":
    main()
    # Run the original test
    test_result = asyncio.run(test_lilith_core())
    print(f"Test result: {test_result}")