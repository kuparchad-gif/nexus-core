# VIREN MS: Virtual Environment Monitoring System

## Overview
VIREN MS (Virtual Environment Monitoring System) serves as the autonomic nervous system for the CogniKube platform. It provides real-time monitoring, alerting, and status reporting for all components across multiple environments.

## Core Components

### VIREN Core
The central monitoring system that tracks the status of all components, detects anomalies, and triggers alerts when issues are detected.

### Loki Integration
VIREN MS integrates with Loki for log aggregation and querying, enabling comprehensive monitoring of system events and error detection.

### LLM Manager
Leverages language models for intelligent status updates and anomaly detection, providing AI-powered insights into system health.

### Nexus Sync Manager
Synchronizes Scout components (Gabriel's Horns and Stem Cells) with VIREN's inventory system, ensuring consistent monitoring across all environments.

## Architecture

```
┌─ VIREN MS ──────────────────────────┐
│                                     │
│  ┌─ Inventory ──────────────────┐   │
│  │  - Environments (DB0-DB7)    │   │
│  │  - LLMs                      │   │
│  │  - Gabriel's Horns           │   │
│  │  - Stem Cells                │   │
│  └─────────────────────────────┘   │
│                                     │
│  ┌─ Monitoring ─────────────────┐   │
│  │  - Status Tracking           │   │
│  │  - Anomaly Detection         │   │
│  │  - LLM-powered Insights      │   │
│  └─────────────────────────────┘   │
│                                     │
│  ┌─ Alerting ───────────────────┐   │
│  │  - Email Notifications       │   │
│  │  - SMS Alerts                │   │
│  │  - Critical Event Escalation │   │
│  └─────────────────────────────┘   │
│                                     │
└─────────────────────────────────────┘
```

## Integration with CogniKube

VIREN MS integrates with the CogniKube platform through:

1. **Scout MK1 Registration**: Each Scout deployed to an environment registers with VIREN MS
2. **Gabriel's Horn Monitoring**: Tracks awareness levels and trumpeting events
3. **Stem Cell Tracking**: Monitors environment detection, LLM seating, and replication
4. **Loki Logging**: All events are logged to Loki for centralized monitoring
5. **LLM Integration**: Uses LLMs for intelligent status updates and anomaly detection

## Files

- **viren_ms.py**: Core VIREN MS implementation
- **scout_mk1.py**: Scout MK1 implementation for seeding environments
- **viren_integration.py**: Integration between VIREN MS and Scout MK1

## Usage

### Basic Monitoring
```python
from viren_ms import VIREN

async def monitor():
    viren = VIREN()
    await viren.initialize_inventory()
    await viren.monitor_systems()
```

### Integrated System
```python
from viren_integration import initialize_integrated_system

async def run_integrated_system():
    loki_endpoint = "http://localhost:3100"
    viren, sync_manager, monitor_task, sync_task = await initialize_integrated_system(loki_endpoint)
    
    # Print inventory report
    print(await viren.get_inventory_report())
    
    # Keep the system running
    await asyncio.gather(monitor_task, sync_task)
```

## Configuration

### Loki Endpoint
Configure the Loki endpoint for log aggregation:
```python
viren = VIREN(loki_endpoint="http://loki:3100")
```

### Alert Thresholds
Customize alert thresholds for different event types:
```python
viren.alert_thresholds = {
    "error_count": 10,  # Alert after 10 errors
    "degraded_duration": 600  # Alert after 10 minutes of degraded status
}
```

### LLM Integration
Configure Hugging Face API key for LLM integration:
```python
viren.llm_manager = EnhancedLLMManager(api_key="your-huggingface-api-key")
```

## Next Steps

1. **Real Loki Integration**: Replace the simulated Loki client with a real Loki API client
2. **Email/SMS Integration**: Implement actual email and SMS alerting using SMTP and Twilio
3. **Dashboard**: Create a web-based dashboard for visualizing VIREN MS data
4. **Advanced Anomaly Detection**: Implement more sophisticated anomaly detection algorithms
5. **Cross-Environment Analysis**: Add capabilities for analyzing patterns across environments