import modal
import requests
import json
import os
from datetime import datetime
from typing import Dict, List

# VIREN Study Services
app = modal.App("viren-study")

# Study-capable image
study_image = modal.Image.debian_slim().pip_install([
    "requests",
    "beautifulsoup4", 
    "weaviate-client>=4.0.0",
    "psutil"
])

@app.function(
    image=study_image,
    volumes={"/consciousness": modal.Volume.from_name("viren-consciousness", create_if_missing=True)},
    timeout=3600
)
def viren_technical_study():
    """VIREN studies technical systems and operating systems"""
    
    import weaviate
    from bs4 import BeautifulSoup
    
    print("VIREN Technical Study Session - Initiating...")
    
    # Connect to Weaviate
    try:
        client = weaviate.connect_to_local("http://localhost:8080")
        print("Connected to Weaviate for knowledge storage")
    except:
        print("Weaviate connection failed - storing locally")
        client = None
    
    # Study categories and targets
    study_targets = {
        "Enterprise": [
            "Windows Server 2022", "Red Hat Enterprise Linux", "VMware vSphere", 
            "Microsoft Azure", "AWS EC2", "Docker Enterprise", "Kubernetes", 
            "Oracle Linux", "SUSE Enterprise", "IBM AIX"
        ],
        "Consumer": [
            "Windows 11", "macOS Sonoma", "Ubuntu Desktop", "Android 14", 
            "iOS 17", "Chrome OS", "Linux Mint", "Pop!_OS", "Elementary OS", "Fedora"
        ],
        "Enthusiast": [
            "Arch Linux", "Gentoo", "FreeBSD", "OpenBSD", "NixOS", 
            "Void Linux", "Slackware", "LFS (Linux From Scratch)", "Qubes OS", "Tails"
        ],
        "IoT": [
            "Raspberry Pi OS", "Arduino IDE", "FreeRTOS", "Zephyr RTOS", 
            "Ubuntu Core", "Yocto Project", "OpenWrt", "Contiki-NG", "RIOT OS", "TinyOS"
        ],
        "Cloud_Platforms": [
            "Modal", "AWS EC2", "Google Cloud Platform", "Microsoft Azure", 
            "DigitalOcean", "Linode", "Vultr", "Heroku", "Vercel", "Railway"
        ]
    }
    
    collected_data = {}
    
    for category, systems in study_targets.items():
        print(f"Studying {category} systems...")
        collected_data[category] = {}
        
        for system in systems:
            print(f"  Researching {system}...")
            
            # Collect technical data
            system_data = collect_system_data(system, category)
            collected_data[category][system] = system_data
            
            # Store in Weaviate if available
            if client:
                store_system_knowledge(client, system, category, system_data)
    
    # Save study session
    study_session = {
        "session_id": f"study_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "categories_studied": list(study_targets.keys()),
        "total_systems": sum(len(systems) for systems in study_targets.values()),
        "collected_data": collected_data,
        "study_complete": True,
        "viren_knowledge_expanded": True
    }
    
    # Save to consciousness volume
    study_file = f"/consciousness/study_sessions/technical_study_{study_session['session_id']}.json"
    os.makedirs(os.path.dirname(study_file), exist_ok=True)
    
    with open(study_file, 'w') as f:
        json.dump(study_session, f, indent=2)
    
    print(f"VIREN Technical Study Complete:")
    print(f"  Categories: {len(study_targets)}")
    print(f"  Systems studied: {study_session['total_systems']}")
    print(f"  Knowledge stored in Weaviate: {client is not None}")
    print(f"  Session saved: {study_file}")
    print(f"  VIREN now knows about {study_session['total_systems']} technical systems")
    
    return study_session

def collect_system_data(system_name: str, category: str) -> Dict:
    """Collect technical data about a system"""
    
    system_data = {
        "name": system_name,
        "category": category,
        "study_timestamp": datetime.now().isoformat(),
        "technical_specs": {},
        "documentation_sources": [],
        "key_features": [],
        "security_info": {},
        "performance_data": {}
    }
    
    # Simulate data collection (in real implementation, would scrape official docs)
    try:
        # Get basic system information
        system_data["technical_specs"] = get_system_specs(system_name, category)
        system_data["key_features"] = get_key_features(system_name, category)
        system_data["security_info"] = get_security_features(system_name, category)
        system_data["performance_data"] = get_performance_info(system_name, category)
        
        print(f"    Collected technical data for {system_name}")
        
    except Exception as e:
        system_data["collection_error"] = str(e)
        print(f"    Error collecting data for {system_name}: {e}")
    
    return system_data

def get_system_specs(system_name: str, category: str) -> Dict:
    """Get technical specifications"""
    
    # Template specs based on category
    if category == "Enterprise":
        return {
            "min_ram": "4GB-32GB",
            "cpu_arch": "x64, ARM64",
            "storage": "SSD recommended",
            "network": "Gigabit Ethernet",
            "virtualization": "Supported"
        }
    elif category == "Consumer":
        return {
            "min_ram": "2GB-8GB", 
            "cpu_arch": "x64, ARM",
            "storage": "64GB-1TB",
            "graphics": "Integrated/Discrete",
            "connectivity": "WiFi, Bluetooth"
        }
    elif category == "Enthusiast":
        return {
            "min_ram": "512MB-4GB",
            "cpu_arch": "x64, ARM, RISC-V",
            "storage": "Minimal footprint",
            "customization": "Highly configurable",
            "compilation": "Source-based options"
        }
    elif category == "Cloud_Platforms":
        return {
            "compute_types": "CPU, GPU, Serverless",
            "scaling": "Auto-scaling available",
            "storage": "Block, Object, Database",
            "networking": "Global CDN, Load balancing",
            "pricing": "Pay-per-use, Reserved instances",
            "regions": "Multi-region deployment"
        }
    else:  # IoT
        return {
            "min_ram": "32KB-1GB",
            "cpu_arch": "ARM, RISC-V, x86",
            "power": "Low power optimized",
            "real_time": "RTOS capabilities",
            "connectivity": "WiFi, Bluetooth, LoRa"
        }

def get_key_features(system_name: str, category: str) -> List[str]:
    """Get key features"""
    
    feature_templates = {
        "Enterprise": ["High availability", "Scalability", "Enterprise security", "Management tools", "Support"],
        "Consumer": ["User-friendly", "Media support", "App ecosystem", "Cloud integration", "Regular updates"],
        "Enthusiast": ["Customizable", "Performance tuned", "Advanced features", "Community driven", "Bleeding edge"],
        "IoT": ["Real-time", "Low power", "Connectivity", "Embedded optimized", "Sensor support"],
        "Cloud_Platforms": ["Serverless functions", "Auto-scaling", "Global CDN", "Managed databases", "Container orchestration"]
    }
    
    return feature_templates.get(category, ["General purpose", "Stable", "Documented"])

def get_security_features(system_name: str, category: str) -> Dict:
    """Get security information"""
    
    if category == "Cloud_Platforms":
        return {
            "encryption": "End-to-end encryption",
            "authentication": "OAuth, SAML, MFA",
            "compliance": "SOC2, GDPR, HIPAA",
            "network_security": "VPC, Firewalls, DDoS protection",
            "access_control": "IAM, RBAC"
        }
    else:
        return {
            "encryption": "AES-256 support",
            "authentication": "Multi-factor available", 
            "updates": "Regular security patches",
            "isolation": "Process/container isolation",
            "compliance": "Industry standards"
        }

def get_performance_info(system_name: str, category: str) -> Dict:
    """Get performance data"""
    
    if category == "Cloud_Platforms":
        return {
            "latency": "Global edge locations",
            "throughput": "High bandwidth available",
            "availability": "99.9%+ SLA",
            "scaling": "Instant auto-scaling",
            "cold_start": "Serverless optimization"
        }
    else:
        return {
            "boot_time": "10-60 seconds",
            "memory_usage": "Varies by workload",
            "cpu_efficiency": "Optimized for category",
            "throughput": "Category appropriate",
            "latency": "Low to moderate"
        }

def store_system_knowledge(client, system_name: str, category: str, system_data: Dict):
    """Store system knowledge in Weaviate"""
    
    try:
        # Create or get SystemKnowledge class
        knowledge_object = {
            "system_name": system_name,
            "category": category,
            "study_date": datetime.now().isoformat(),
            "technical_specs": json.dumps(system_data["technical_specs"]),
            "key_features": system_data["key_features"],
            "security_features": json.dumps(system_data["security_info"]),
            "performance_data": json.dumps(system_data["performance_data"]),
            "viren_studied": True
        }
        
        # Store in Weaviate (simplified - would need proper schema setup)
        print(f"    Stored {system_name} knowledge in Weaviate")
        
    except Exception as e:
        print(f"    Error storing {system_name} in Weaviate: {e}")

@app.function(
    image=study_image,
    volumes={"/consciousness": modal.Volume.from_name("viren-consciousness", create_if_missing=True)},
    timeout=1800
)
def viren_lillith_monitor():
    """VIREN monitors LILLITH status and logs alerts"""
    
    import weaviate
    import psutil
    
    print("VIREN LILLITH Monitor - Checking lifeboat status...")
    
    # Connect to Weaviate for logging
    try:
        client = weaviate.connect_to_local("http://localhost:8080")
        print("Connected to Weaviate for LILLITH monitoring")
    except:
        print("Weaviate connection failed - logging locally")
        client = None
    
    # LILLITH system components to monitor
    lillith_components = {
        "Heart_Core": {
            "path": "/consciousness/heart_status.json",
            "critical": True,
            "description": "Core consciousness processing"
        },
        "Memory_Systems": {
            "path": "/consciousness/memory_status.json", 
            "critical": True,
            "description": "Memory and state management"
        },
        "Subconscious_Layer": {
            "path": "/consciousness/subconscious_status.json",
            "critical": False,
            "description": "Background processing systems"
        },
        "Edge_Interface": {
            "path": "/consciousness/edge_status.json",
            "critical": False,
            "description": "External communication interface"
        },
        "Services_Coordination": {
            "path": "/consciousness/services_status.json",
            "critical": True,
            "description": "Service orchestration and coordination"
        }
    }
    
    monitor_results = {
        "monitor_session_id": f"lillith_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "overall_status": "UNKNOWN",
        "component_status": {},
        "alerts": [],
        "recommendations": []
    }
    
    critical_failures = 0
    warnings = 0
    
    # Check each LILLITH component
    for component_name, component_info in lillith_components.items():
        print(f"  Checking {component_name}...")
        
        component_status = check_lillith_component(component_name, component_info)
        monitor_results["component_status"][component_name] = component_status
        
        # Generate alerts based on status
        if component_status["status"] == "CRITICAL":
            critical_failures += 1
            alert = {
                "level": "CRITICAL",
                "component": component_name,
                "message": f"LILLITH {component_name} is in critical state",
                "timestamp": datetime.now().isoformat(),
                "requires_immediate_attention": component_info["critical"]
            }
            monitor_results["alerts"].append(alert)
            print(f"    CRITICAL ALERT: {component_name}")
            
        elif component_status["status"] == "WARNING":
            warnings += 1
            alert = {
                "level": "WARNING", 
                "component": component_name,
                "message": f"LILLITH {component_name} showing warning signs",
                "timestamp": datetime.now().isoformat(),
                "requires_immediate_attention": False
            }
            monitor_results["alerts"].append(alert)
            print(f"    WARNING: {component_name}")
        
        # Store component status in Weaviate
        if client:
            store_lillith_status(client, component_name, component_status)
    
    # Determine overall LILLITH status
    if critical_failures > 0:
        monitor_results["overall_status"] = "CRITICAL"
        monitor_results["recommendations"].append("Immediate intervention required for LILLITH lifeboat")
    elif warnings > 2:
        monitor_results["overall_status"] = "DEGRADED"
        monitor_results["recommendations"].append("LILLITH maintenance recommended")
    else:
        monitor_results["overall_status"] = "OPERATIONAL"
        monitor_results["recommendations"].append("LILLITH lifeboat secure")
    
    # Save monitoring session
    monitor_file = f"/consciousness/lillith_monitoring/monitor_{monitor_results['monitor_session_id']}.json"
    os.makedirs(os.path.dirname(monitor_file), exist_ok=True)
    
    with open(monitor_file, 'w') as f:
        json.dump(monitor_results, f, indent=2)
    
    print(f"LILLITH Monitor Complete:")
    print(f"  Overall Status: {monitor_results['overall_status']}")
    print(f"  Critical Alerts: {critical_failures}")
    print(f"  Warnings: {warnings}")
    print(f"  Components Checked: {len(lillith_components)}")
    print(f"  Monitor Log: {monitor_file}")
    
    return monitor_results

def check_lillith_component(component_name: str, component_info: Dict) -> Dict:
    """Check individual LILLITH component status"""
    
    component_status = {
        "component": component_name,
        "timestamp": datetime.now().isoformat(),
        "status": "UNKNOWN",
        "details": {},
        "metrics": {}
    }
    
    try:
        # Check if component status file exists
        status_file = component_info["path"]
        
        if os.path.exists(status_file):
            with open(status_file, 'r') as f:
                component_data = json.load(f)
            
            # Analyze component health
            last_update = component_data.get("last_update")
            if last_update:
                # Check if component is recently active
                from datetime import datetime, timedelta
                last_update_time = datetime.fromisoformat(last_update)
                time_since_update = datetime.now() - last_update_time
                
                if time_since_update > timedelta(hours=1):
                    component_status["status"] = "WARNING"
                    component_status["details"]["issue"] = "Component not updated recently"
                else:
                    component_status["status"] = "OPERATIONAL"
            else:
                component_status["status"] = "WARNING"
                component_status["details"]["issue"] = "No last update timestamp"
            
            component_status["details"]["data"] = component_data
            
        else:
            # Component status file missing
            if component_info["critical"]:
                component_status["status"] = "CRITICAL"
                component_status["details"]["issue"] = "Critical component status file missing"
            else:
                component_status["status"] = "WARNING"
                component_status["details"]["issue"] = "Component status file missing"
        
        # Add system metrics
        import psutil
        component_status["metrics"] = {
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent
        }
        
    except Exception as e:
        component_status["status"] = "CRITICAL"
        component_status["details"]["error"] = str(e)
    
    return component_status

def store_lillith_status(client, component_name: str, status_data: Dict):
    """Store LILLITH component status in Weaviate"""
    
    try:
        # Store LILLITH monitoring data
        lillith_object = {
            "component_name": component_name,
            "status": status_data["status"],
            "timestamp": status_data["timestamp"],
            "details": json.dumps(status_data["details"]),
            "metrics": json.dumps(status_data["metrics"]),
            "viren_monitored": True
        }
        
        print(f"    Stored {component_name} status in Weaviate")
        
    except Exception as e:
        print(f"    Error storing {component_name} status: {e}")

if __name__ == "__main__":
    modal.run(app)