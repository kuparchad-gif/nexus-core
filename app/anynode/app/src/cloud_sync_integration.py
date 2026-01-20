#!/usr/bin/env python
"""
VIREN Cloud Sync Integration
Connects new awareness system with existing cloud infrastructure
"""

import sys
import os
from pathlib import Path

def sync_awareness_to_cloud():
    """Sync VIREN awareness data to cloud using existing infrastructure"""
    try:
        # Add existing Viren cloud path
        sys.path.insert(0, "C:/Viren")
        from weaviate_sync import sync_local_to_cloud
        
        print("☁️ Syncing VIREN awareness to cloud...")
        sync_local_to_cloud()
        print("✅ Cloud sync complete")
        return True
        
    except Exception as e:
        print(f"⚠️ Cloud sync failed: {e}")
        return False

def sync_from_cloud():
    """Pull data from cloud VIREN instances"""
    try:
        sys.path.insert(0, "C:/Viren")
        from weaviate_sync import sync_cloud_to_local
        
        print("☁️ Pulling data from cloud VIREN...")
        sync_cloud_to_local()
        print("✅ Cloud pull complete")
        return True
        
    except Exception as e:
        print(f"⚠️ Cloud pull failed: {e}")
        return False

def bidirectional_sync():
    """Run bidirectional sync with cloud"""
    print("🔄 Starting bidirectional cloud sync...")
    
    # Pull from cloud first
    sync_from_cloud()
    
    # Push to cloud
    sync_awareness_to_cloud()
    
    print("🔄 Bidirectional sync complete")

if __name__ == "__main__":
    bidirectional_sync()