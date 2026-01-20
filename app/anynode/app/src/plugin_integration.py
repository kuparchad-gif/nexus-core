#!/usr/bin/env python3
"""
Plugin Integration for Cloud Viren
Integrates Desktop Viren plugins into Cloud Viren
"""

import os
import sys
import shutil
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PluginIntegration")

class PluginIntegration:
    """Integrates Desktop Viren plugins into Cloud Viren"""
    
    def __init__(self, desktop_plugins_dir=None, cloud_plugins_dir=None):
        """Initialize the plugin integration"""
        self.desktop_plugins_dir = desktop_plugins_dir or os.path.join('C:/Engineers/root/app/mcp_utils')
        self.cloud_plugins_dir = cloud_plugins_dir or os.path.join('C:/Viren/cloud/Cloud UI Plugins')
    
    def sync_plugins(self):
        """Sync plugins from Desktop Viren to Cloud Viren"""
        try:
            # Check if directories exist
            if not os.path.exists(self.desktop_plugins_dir):
                logger.error(f"Desktop plugins directory not found: {self.desktop_plugins_dir}")
                return False
            
            # Create cloud plugins directory if it doesn't exist
            os.makedirs(self.cloud_plugins_dir, exist_ok=True)
            
            # Get list of plugins in desktop directory
            desktop_plugins = [item for item in os.listdir(self.desktop_plugins_dir) 
                              if os.path.isdir(os.path.join(self.desktop_plugins_dir, item))]
            
            # Get list of plugins in cloud directory
            cloud_plugins = [item for item in os.listdir(self.cloud_plugins_dir) 
                            if os.path.isdir(os.path.join(self.cloud_plugins_dir, item))]
            
            # Copy utility files
            utility_files = ['database.py', 'document_tools.py', 'memory.py', 
                           'module_Scanning.py', 'modules.py', 'system_scan.py', 'voice.py']
            
            for file in utility_files:
                src_file = os.path.join(self.desktop_plugins_dir, file)
                dst_file = os.path.join(self.cloud_plugins_dir, file)
                
                if os.path.exists(src_file):
                    shutil.copy2(src_file, dst_file)
                    logger.info(f"Copied utility file: {file}")
            
            # Copy missing plugins
            for plugin in desktop_plugins:
                # Skip utility files
                if plugin in utility_files:
                    continue
                
                src_dir = os.path.join(self.desktop_plugins_dir, plugin)
                dst_dir = os.path.join(self.cloud_plugins_dir, plugin)
                
                # Check if plugin is a directory
                if not os.path.isdir(src_dir):
                    continue
                
                # Check if plugin already exists in cloud
                if plugin in cloud_plugins:
                    # Check if desktop version is newer
                    src_mtime = os.path.getmtime(src_dir)
                    dst_mtime = os.path.getmtime(dst_dir)
                    
                    if src_mtime > dst_mtime:
                        # Remove existing plugin
                        shutil.rmtree(dst_dir)
                        # Copy plugin
                        shutil.copytree(src_dir, dst_dir)
                        logger.info(f"Updated plugin: {plugin}")
                else:
                    # Copy plugin
                    shutil.copytree(src_dir, dst_dir)
                    logger.info(f"Copied plugin: {plugin}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error syncing plugins: {e}")
            return False
    
    def copy_orb_assets(self):
        """Copy orb assets from Desktop Viren to Cloud Viren"""
        try:
            # Source files
            src_orb_video = os.path.join('C:/Engineers/root/app', 'morph_orb.mp4')
            src_orb_image = os.path.join('C:/Engineers/root/app', 'orb.png')
            src_style_css = os.path.join('C:/Engineers/root/app', 'style.css')
            src_orb_app = os.path.join('C:/Engineers/root/app', 'orb_gradio_app.py')
            
            # Destination directory
            dst_dir = os.path.join('C:/Viren/cloud/assets')
            os.makedirs(dst_dir, exist_ok=True)
            
            # Copy files
            if os.path.exists(src_orb_video):
                shutil.copy2(src_orb_video, os.path.join(dst_dir, 'morph_orb.mp4'))
                logger.info(f"Copied orb video")
            
            if os.path.exists(src_orb_image):
                shutil.copy2(src_orb_image, os.path.join(dst_dir, 'orb.png'))
                logger.info(f"Copied orb image")
            
            if os.path.exists(src_style_css):
                shutil.copy2(src_style_css, os.path.join(dst_dir, 'style.css'))
                logger.info(f"Copied style.css")
            
            if os.path.exists(src_orb_app):
                shutil.copy2(src_orb_app, os.path.join(dst_dir, 'orb_gradio_app.py'))
                logger.info(f"Copied orb app")
            
            return True
        
        except Exception as e:
            logger.error(f"Error copying orb assets: {e}")
            return False
    
    def copy_config_files(self):
        """Copy configuration files from Desktop Viren to Cloud Viren"""
        try:
            # Source files
            src_config_dir = os.path.join('C:/Engineers/root/Config')
            
            # Destination directory
            dst_config_dir = os.path.join('C:/Viren/config')
            os.makedirs(dst_config_dir, exist_ok=True)
            
            # Copy important config files
            config_files = [
                'viren_identity.py',
                'viren_soulprint.json',
                'model_config.py',
                'model_config.json'
            ]
            
            for file in config_files:
                src_file = os.path.join(src_config_dir, file)
                dst_file = os.path.join(dst_config_dir, file)
                
                if os.path.exists(src_file):
                    shutil.copy2(src_file, dst_file)
                    logger.info(f"Copied config file: {file}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error copying config files: {e}")
            return False
    
    def integrate_all(self):
        """Integrate all Desktop Viren components into Cloud Viren"""
        logger.info("Starting plugin integration")
        
        # Create directories
        os.makedirs(os.path.join('C:/Viren/cloud/assets'), exist_ok=True)
        os.makedirs(os.path.join('C:/Viren/cloud/Cloud UI Plugins'), exist_ok=True)
        
        # Sync plugins
        self.sync_plugins()
        
        # Copy orb assets
        self.copy_orb_assets()
        
        # Copy config files
        self.copy_config_files()
        
        logger.info("Plugin integration complete")
        return True

def main():
    """Main entry point"""
    integration = PluginIntegration()
    integration.integrate_all()

if __name__ == "__main__":
    main()