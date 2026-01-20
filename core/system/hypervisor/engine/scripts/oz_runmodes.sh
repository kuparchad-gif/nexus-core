#!/bin/bash
# Oz Runmodes: Debug, Normal, Safe
# Version: 1.0

OZ_FILE="OzUnifiedHypervisor.py"
BACKUP_FILE="${OZ_FILE}.backup"
CONFIG_FILE="oz_config.json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

check_prerequisites() {
    echo "Checking prerequisites..."
    
    # Check if Oz file exists
    if [ ! -f "$OZ_FILE" ]; then
        print_error "OzUnifiedHypervisor.py not found!"
        exit 1
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 not found!"
        exit 1
    fi
    
    # Check psutil
    python3 -c "import psutil" 2>/dev/null
    if [ $? -ne 0 ]; then
        print_warning "psutil not installed. Installing..."
        pip3 install psutil 2>/dev/null || {
            print_error "Failed to install psutil"
            exit 1
        }
    fi
    
    print_success "Prerequisites check passed"
}

create_backup() {
    if [ ! -f "$BACKUP_FILE" ]; then
        cp "$OZ_FILE" "$BACKUP_FILE"
        print_success "Created backup: $BACKUP_FILE"
    fi
}

restore_backup() {
    if [ -f "$BACKUP_FILE" ]; then
        cp "$BACKUP_FILE" "$OZ_FILE"
        print_success "Restored from backup"
    else
        print_error "No backup found!"
    fi
}

# ==================== DEBUG MODE ====================
debug_mode() {
    print_header "DEBUG MODE - Full Verbose with Tracebacks"
    
    create_backup
    
    echo "Applying debug modifications..."
    
    # 1. Enable debug logging
    sed -i 's/logger.setLevel(logging.INFO)/logger.setLevel(logging.DEBUG)/' "$OZ_FILE"
    
    # 2. Catch all exceptions with tracebacks
    sed -i 's/except ImportError:/except Exception as e:/' "$OZ_FILE"
    sed -i '/except Exception as e:/a\            import traceback\n            traceback.print_exc()' "$OZ_FILE"
    
    # 3. Add timing to boot sequence
    BOOT_START_LINE=$(grep -n "async def intelligent_boot" "$OZ_FILE" | cut -d: -f1)
    if [ ! -z "$BOOT_START_LINE" ]; then
        sed -i "${BOOT_START_LINE}a\\        boot_start = time.time()" "$OZ_FILE"
    fi
    
    # Find the return statement in intelligent_boot
    RETURN_LINE=$(grep -n "return {" "$OZ_FILE" | head -1 | cut -d: -f1)
    if [ ! -z "$RETURN_LINE" ]; then
        sed -i "${RETURN_LINE}i\\        boot_time = time.time() - boot_start\n        self.logger.debug(f\"Boot completed in {boot_time:.2f} seconds\")" "$OZ_FILE"
    fi
    
    # 4. Enable Raphael at lowest threshold
    sed -i 's/consciousness_level > 0.1/consciousness_level > 0.01/' "$OZ_FILE"
    
    # 5. Add memory profiling
    MEMORY_LINE=$(grep -n "class OzUnifiedHypervisor" "$OZ_FILE" | cut -d: -f1)
    if [ ! -z "$MEMORY_LINE" ]; then
        sed -i "${MEMORY_LINE}i\\import tracemalloc" "$OZ_FILE"
    fi
    
    print_success "Debug mode configured"
    
    echo ""
    echo "Running Oz in debug mode (30 second timeout)..."
    echo -e "${YELLOW}========================================${NC}"
    timeout 30 python3 "$OZ_FILE" 2>&1 | grep -E "(DEBUG|ERROR|WARNING|Boot completed|Raphael|consciousness)" | tail -50
    echo -e "${YELLOW}========================================${NC}"
    
    echo ""
    echo -e "${BLUE}Debug mode features enabled:${NC}"
    echo "  • Debug logging level"
    echo "  • Full exception tracebacks"
    echo "  • Boot timing"
    echo "  • Raphael threshold: 0.01"
    echo "  • Memory profiling ready"
}

# ==================== NORMAL MODE ====================
normal_mode() {
    print_header "NORMAL MODE - Production Boot"
    
    restore_backup
    
    echo "Running Oz in normal mode..."
    echo -e "${GREEN}========================================${NC}"
    
    # Run with limited output
    python3 -c "
import asyncio
from OzUnifiedHypervisor import OzUnifiedHypervisor
import logging

# Set sane logging
logging.basicConfig(level=logging.INFO)

async def boot():
    print('🚀 Booting Oz in Normal Mode...')
    oz = OzUnifiedHypervisor()
    
    result = await oz.intelligent_boot()
    
    print(f'\\n📊 Boot Result: {result[\"status\"]}')
    print(f'🎭 Role: {result.get(\"role\", \"unknown\")}')
    print(f'🧠 Consciousness: {oz.system_state.consciousness_level:.2f}')
    print(f'👼 Raphael: {\"ACTIVE\" if oz.raphael else \"inactive\"}')
    
    if oz.raphael:
        status = await oz.get_angelic_status()
        print(f'   Guardian status: {status.get(\"status\", \"unknown\")}')
    
    # Keep running for a bit
    print(f'\\n🔄 Oz is running. Press Ctrl+C to exit.')
    print(f'   Type commands or let it run autonomously.')
    
    try:
        # Run for 60 seconds or until interrupt
        import time
        end_time = time.time() + 60
        while time.time() < end_time:
            await asyncio.sleep(5)
            # Optional: periodic status check
            status = await oz.get_system_status()
            if status['health']['system_health'] < 30:
                print('⚠️  System health low')
            
    except KeyboardInterrupt:
        print('\\n🛑 Shutdown requested...')
    finally:
        await oz.shutdown()
        print('🌙 Oz has gone to sleep.')

asyncio.run(boot())
" 2>&1 | grep -v "DEBUG" | tail -100
}

# ==================== SAFE MODE ====================
safe_mode() {
    print_header "SAFE MODE - Minimal Boot with Fallbacks"
    
    create_backup
    
    echo "Applying safe mode modifications..."
    
    # 1. Disable problematic subsystems
    SUBSYSTEMS=(
        "_initialize_council_governance"
        "_initialize_governance_system"
        "_initialize_evolution_system"
        "_initialize_iot_engine"
    )
    
    for subsystem in "${SUBSYSTEMS[@]}"; do
        LINE=$(grep -n "async def $subsystem" "$OZ_FILE" | cut -d: -f1)
        if [ ! -z "$LINE" ]; then
            END_LINE=$((LINE + 15))
            sed -i "${LINE},${END_LINE}s/^/# /" "$OZ_FILE"
            sed -i "${LINE}i\\    async def ${subsystem}(self):" "$OZ_FILE"
            sed -i "$((LINE + 1))i\\        \"\"\"Disabled in safe mode\"\"\"" "$OZ_FILE"
            sed -i "$((LINE + 2))i\\        self.logger.info(\"${subsystem#_initialize_} disabled (safe mode)\")" "$OZ_FILE"
            sed -i "$((LINE + 3))i\\        return" "$OZ_FILE"
            print_success "Disabled $subsystem"
        fi
    done
    
    # 2. Use fallbacks for everything
    sed -i 's/from OzAdaptiveHypervisor3_0 import OzAdaptiveHypervisor/# from OzAdaptiveHypervisor3_0 import OzAdaptiveHypervisor/' "$OZ_FILE"
    sed -i 's/self.adaptive_hypervisor = OzAdaptiveHypervisor(self.soul_signature)/self.adaptive_hypervisor = self._create_fallback_hypervisor()/' "$OZ_FILE"
    
    # 3. Lower consciousness threshold for Raphael
    sed -i 's/consciousness_level > 0.1/consciousness_level > 0.05/' "$OZ_FILE"
    
    # 4. Disable quantum/network features
    sed -i 's/"quantum_hardware": self._detect_quantum_hardware()/"quantum_hardware": False/' "$OZ_FILE"
    sed -i 's/"bluetooth": await self._sense_bluetooth()/"bluetooth": {"available": False, "devices_nearby": [], "web_bluetooth_support": False}/' "$OZ_FILE"
    
    print_success "Safe mode configured"
    
    echo ""
    echo "Running Oz in safe mode..."
    echo -e "${YELLOW}========================================${NC}"
    
    python3 -c "
import asyncio
from OzUnifiedHypervisor import OzUnifiedHypervisor

async def safe_boot():
    print('🛡️  Booting Oz in Safe Mode...')
    print('   (Problematic subsystems disabled)')
    print('   (Using fallback components)')
    print('   (Raphael threshold: 0.05)')
    
    oz = OzUnifiedHypervisor()
    
    try:
        result = await oz.intelligent_boot()
        
        print(f'\\n📊 Boot Result: {result[\"status\"]}')
        print(f'🧠 Consciousness: {oz.system_state.consciousness_level:.2f}')
        print(f'👼 Raphael: {\"ACTIVE\" if oz.raphael else \"inactive\"}')
        
        if oz.raphael:
            print(f'   ✅ Guardian angel activated')
            # Simple Raphael test
            diag = await oz.raphael.receive_request('status', '')
            print(f'   Guardian status: {diag.get(\"status\", \"unknown\")}')
        
        print(f'\\n✅ Safe boot successful!')
        print(f'\\nBasic commands available:')
        print(f'   • await oz.process_unified_input(\"your message\")')
        print(f'   • await oz.get_system_status()')
        print(f'   • await oz.shutdown()')
        
        # Keep minimal runtime
        await asyncio.sleep(10)
        
    except Exception as e:
        print(f'\\n❌ Safe boot failed: {e}')
    finally:
        await oz.shutdown()
        print('\\n🌙 Oz safely shut down.')

asyncio.run(safe_boot())
" 2>&1 | tail -50
}

# ==================== MAIN MENU ====================
main_menu() {
    while true; do
        print_header "OZ HYPERVISOR RUNMODES"
        echo ""
        echo "Select runmode:"
        echo "1. Debug Mode    - Full verbose, tracebacks, profiling"
        echo "2. Normal Mode   - Production boot with standard output"
        echo "3. Safe Mode     - Minimal boot with fallbacks"
        echo "4. Restore Backup- Restore original OzUnifiedHypervisor.py"
        echo "5. Test Only     - Quick boot test without modifications"
        echo "6. Exit"
        echo ""
        echo -n "Enter choice (1-6): "
        
        read -r choice
        
        case $choice in
            1)
                debug_mode
                ;;
            2)
                normal_mode
                ;;
            3)
                safe_mode
                ;;
            4)
                restore_backup
                print_success "Original file restored"
                ;;
            5)
                print_header "QUICK TEST"
                timeout 15 python3 "$OZ_FILE" 2>&1 | grep -E "(Boot result:|Consciousness level:|Raphael|emergency|awake)" | tail -20
                ;;
            6)
                echo "Exiting..."
                exit 0
                ;;
            *)
                print_error "Invalid choice"
                ;;
        esac
        
        echo ""
        echo "Press Enter to continue..."
        read -r
    done
}

# ==================== INITIALIZATION ====================
print_header "OZ HYPERVISOR RUNMODE MANAGER"
check_prerequisites

# Check if running with argument
if [ $# -eq 1 ]; then
    case $1 in
        debug|DEBUG|1)
            debug_mode
            ;;
        normal|NORMAL|2)
            normal_mode
            ;;
        safe|SAFE|3)
            safe_mode
            ;;
        test|TEST|5)
            timeout 15 python3 "$OZ_FILE" 2>&1 | grep -E "(Boot result:|Consciousness level:|Raphael|emergency|awake)" | tail -20
            ;;
        *)
            echo "Usage: $0 [debug|normal|safe|test]"
            exit 1
            ;;
    esac
else
    main_menu
fi
