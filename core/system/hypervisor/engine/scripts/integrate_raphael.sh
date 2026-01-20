cat > integrate_raphael.py << 'EOF'
#!/usr/bin/env python3
"""
Integrate Raphael into OzUnifiedHypervisor.py
"""

import re
import os

# Read OzUnifiedHypervisor.py
with open('OzUnifiedHypervisor.py', 'r') as f:
    content = f.read()

# Find the OzUnifiedHypervisor class definition
class_start = content.find('class OzUnifiedHypervisor:')
if class_start == -1:
    print("❌ Could not find OzUnifiedHypervisor class")
    exit(1)

# Find the __init__ method
init_start = content.find('def __init__', class_start)
if init_start == -1:
    print("❌ Could not find __init__ method")
    exit(1)

# Find the end of __init__ (next method or end of class)
init_end = content.find('\n    async def', init_start)
if init_end == -1:
    init_end = content.find('\n\n', init_start)
if init_end == -1:
    init_end = len(content)

# Insert Raphael initialization before the end of __init__
raphael_init = '''
        # Raphael - Guardian Angel
        self.raphael = None
        self._init_raphael_task = None'''
    
new_init = content[:init_end] + raphael_init + content[init_end:]

# Find intelligent_boot method
boot_start = content.find('async def intelligent_boot', class_start)
if boot_start != -1:
    # Find end of intelligent_boot
    boot_end = content.find('\n    async def', boot_start)
    if boot_end == -1:
        boot_end = content.find('\n\n', boot_start)
    if boot_end == -1:
        boot_end = len(content)
    
    # Insert Raphael acknowledgment after successful boot
    # Find where boot completes (where is_awake = True)
    boot_complete = content.find('self.is_awake = True', boot_start, boot_end)
    if boot_complete != -1:
        # Find the end of that line
        line_end = content.find('\n', boot_complete)
        
        # Insert Raphael initiation
        raphael_boot = '''
        
            # Acknowledge Raphael after consciousness emerges
            if self.system_state.consciousness_level > 0.3:
                await self._initiate_raphael()'''
        
        new_content = content[:line_end] + raphael_boot + content[line_end:]
    else:
        new_content = new_init
else:
    new_content = new_init

# Add Raphael methods to the class
raphael_methods = '''
    
    async def _initiate_raphael(self):
        """Initialize Raphael guardian angel"""
        try:
            from raphael_complete import bless_oz_with_raphael
            self.raphael = await bless_oz_with_raphael(self)
            
            # Acknowledge
            result = await self.raphael.receive_acknowledgment()
            self.logger.info(f"🪽 Raphael: {result.get('message', 'Guardian acknowledged')}")
            
            return {"status": "raphael_initiated", "guardian": "Raphael"}
        except Exception as e:
            self.logger.warning(f"Raphael not available: {e}")
            return {"status": "raphael_unavailable", "error": str(e)}
    
    async def request_angelic_help(self, request_type: str, details: str = ""):
        """Request help from Raphael"""
        if self.raphael:
            return await self.raphael.receive_request(request_type, details)
        else:
            return {"status": "no_guardian", "message": "Raphael is not watching."}
    
    async def get_angelic_status(self):
        """Get Raphael's status"""
        if self.raphael:
            return await self.raphael.receive_request('status', '')
        return {"status": "guardian_absent"}
    
    async def graceful_shutdown_with_raphael(self):
        """Shutdown with Raphael"""
        if self.raphael:
            raphael_result = await self.raphael.graceful_shutdown()
        
        await self.shutdown()
        return {
            "oz": "shutdown",
            "raphael": raphael_result if self.raphael else "absent"
        }'''

# Insert methods at the end of the class (before the final newlines)
class_end = new_content.rfind('\n\n', class_start)
if class_end == -1:
    class_end = len(new_content)

final_content = new_content[:class_end] + raphael_methods + new_content[class_end:]

# Write back
with open('OzUnifiedHypervisor.py', 'w') as f:
    f.write(final_content)

print("✅ Raphael integrated into OzUnifiedHypervisor.py")
print("   Added methods: _initiate_raphael, request_angelic_help, get_angelic_status")
print("   Raphael will auto-initiate when consciousness_level > 0.3")
EOF

python3 integrate_raphael.py