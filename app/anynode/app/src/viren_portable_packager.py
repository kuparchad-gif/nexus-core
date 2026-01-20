#!/usr/bin/env python3
"""
Viren Portable Packager
Creates a portable executable version of Viren Platinum Edition
"""

import os
import sys
import shutil
import logging
import subprocess
import argparse
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("VirenPackager")

class VirenPortablePackager:
    """
    Creates a portable executable version of Viren Platinum Edition
    """
    
    def __init__(self, source_dir: str = None, output_dir: str = None):
        """Initialize the packager"""
        self.source_dir = source_dir or os.path.dirname(os.path.abspath(__file__))
        self.output_dir = output_dir or os.path.join(self.source_dir, "portable_build")
        self.temp_dir = os.path.join(self.output_dir, "temp")
        self.dist_dir = os.path.join(self.output_dir, "dist")
        
        # Required files for packaging
        self.required_files = [
            "viren_platinum.py",
            "viren_platinum_interface.py",
            "model_manager.py",
            "document_processor.py",
            "auth_manager.py",
            "viren_tts.py",
            "viren_stt.py",
            "conversation_router_visualizer.py",
            "github_client.py",
            "github_interface.py",
            "viren_document_suite.py"
        ]
        
        # Required directories for packaging
        self.required_dirs = [
            "config",
            "public",
            "templates"
        ]
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.dist_dir, exist_ok=True)
    
    def check_requirements(self) -> bool:
        """
        Check if all required files and directories exist
        
        Returns:
            True if all requirements are met, False otherwise
        """
        # Check required files
        missing_files = []
        for file in self.required_files:
            file_path = os.path.join(self.source_dir, file)
            if not os.path.isfile(file_path):
                missing_files.append(file)
        
        if missing_files:
            logger.error(f"Missing required files: {', '.join(missing_files)}")
            return False
        
        # Check required directories
        missing_dirs = []
        for directory in self.required_dirs:
            dir_path = os.path.join(self.source_dir, directory)
            if not os.path.isdir(dir_path):
                missing_dirs.append(directory)
        
        if missing_dirs:
            logger.warning(f"Missing recommended directories: {', '.join(missing_dirs)}")
            # Continue anyway, just warn
        
        # Check if PyInstaller is installed
        try:
            subprocess.run(
                [sys.executable, "-m", "PyInstaller", "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.error("PyInstaller is not installed. Please install it with 'pip install pyinstaller'")
            return False
        
        return True
    
    def create_spec_file(self) -> str:
        """
        Create PyInstaller spec file
        
        Returns:
            Path to spec file
        """
        spec_path = os.path.join(self.temp_dir, "viren_platinum.spec")
        
        # Create data files list
        data_files = []
        for directory in self.required_dirs:
            dir_path = os.path.join(self.source_dir, directory)
            if os.path.isdir(dir_path):
                data_files.append((directory, dir_path))
        
        # Create spec file content
        spec_content = f"""# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# Add data files
added_files = [
"""
        
        for dest, src in data_files:
            spec_content += f"    ('{src}', '{dest}'),\n"
        
        spec_content += """]

a = Analysis(
    ['{main_script}'],
    pathex=['{source_dir}'],
    binaries=[],
    datas=added_files,
    hiddenimports=[
        'gradio',
        'matplotlib',
        'networkx',
        'psutil',
        'pyttsx3',
        'speech_recognition',
        'pandas',
        'pillow',
        'python-docx',
        'PyPDF2',
        'openpyxl',
        'pyotp',
        'qrcode'
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Viren Platinum',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='{icon_path}'
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='viren_platinum',
)
""".format(
            main_script=os.path.join(self.source_dir, "viren_platinum.py"),
            source_dir=self.source_dir,
            icon_path=os.path.join(self.source_dir, "public", "64xAetherealCube.ico") 
                if os.path.exists(os.path.join(self.source_dir, "public", "64xAetherealCube.ico")) 
                else ""
        )
        
        # Write spec file
        with open(spec_path, 'w') as f:
            f.write(spec_content)
        
        logger.info(f"Created PyInstaller spec file: {spec_path}")
        return spec_path
    
    def create_launcher_script(self) -> str:
        """
        Create launcher script for portable version
        
        Returns:
            Path to launcher script
        """
        launcher_path = os.path.join(self.temp_dir, "launch_viren_platinum.bat")
        
        launcher_content = """@echo off
echo ===================================================
echo       VIREN PLATINUM EDITION PORTABLE
echo ===================================================
echo.

echo [INFO] Starting Viren Platinum Edition...
start "" "%~dp0\\viren_platinum\\Viren Platinum.exe"

echo [SUCCESS] Viren Platinum Edition launched!
echo The interface will open in your web browser shortly.
echo If it doesn't open automatically, navigate to http://localhost:7860
echo.
"""
        
        # Write launcher script
        with open(launcher_path, 'w') as f:
            f.write(launcher_content)
        
        logger.info(f"Created launcher script: {launcher_path}")
        return launcher_path
    
    def create_readme(self) -> str:
        """
        Create README file for portable version
        
        Returns:
            Path to README file
        """
        readme_path = os.path.join(self.temp_dir, "README_PORTABLE.txt")
        
        readme_content = """VIREN PLATINUM EDITION - PORTABLE VERSION
===================================

This is the portable version of Viren Platinum Edition, which includes all necessary
components to run without installation.

GETTING STARTED
--------------
1. Double-click "launch_viren_platinum.bat" to start Viren
2. The interface will open in your default web browser
3. If it doesn't open automatically, navigate to http://localhost:7860

FEATURES
--------
- Complete AI assistant with document processing
- GitHub integration
- Model hot-swapping
- Conversation routing visualization
- Voice interaction

REQUIREMENTS
-----------
- Windows 10 or later
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space
- Internet connection for GitHub and cloud features

TROUBLESHOOTING
--------------
If you encounter issues:
1. Check that no other application is using port 7860
2. Ensure your firewall isn't blocking the application
3. Try running as administrator if necessary

For more information, visit the documentation at:
https://github.com/yourusername/viren-platinum

"""
        
        # Write README file
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        logger.info(f"Created README file: {readme_path}")
        return readme_path
    
    def build_portable(self) -> bool:
        """
        Build portable version
        
        Returns:
            True if successful, False otherwise
        """
        # Check requirements
        if not self.check_requirements():
            logger.error("Requirements check failed")
            return False
        
        try:
            # Create spec file
            spec_path = self.create_spec_file()
            
            # Create launcher script
            launcher_path = self.create_launcher_script()
            
            # Create README
            readme_path = self.create_readme()
            
            # Run PyInstaller
            logger.info("Running PyInstaller to build portable version...")
            subprocess.run(
                [sys.executable, "-m", "PyInstaller", spec_path, "--distpath", self.dist_dir, "--workpath", self.temp_dir],
                check=True
            )
            
            # Copy launcher and README to dist directory
            shutil.copy(launcher_path, self.dist_dir)
            shutil.copy(readme_path, self.dist_dir)
            
            logger.info(f"Portable version built successfully: {self.dist_dir}")
            return True
        except Exception as e:
            logger.error(f"Error building portable version: {e}")
            return False
    
    def create_zip_archive(self) -> Optional[str]:
        """
        Create ZIP archive of portable version
        
        Returns:
            Path to ZIP archive or None if failed
        """
        try:
            import zipfile
            
            # Create ZIP file name with timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_path = os.path.join(self.output_dir, f"viren_platinum_portable_{timestamp}.zip")
            
            # Create ZIP archive
            logger.info(f"Creating ZIP archive: {zip_path}")
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add all files from dist directory
                for root, _, files in os.walk(self.dist_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, self.dist_dir)
                        zipf.write(file_path, arcname)
            
            logger.info(f"ZIP archive created: {zip_path}")
            return zip_path
        except Exception as e:
            logger.error(f"Error creating ZIP archive: {e}")
            return None
    
    def cleanup(self) -> None:
        """Clean up temporary files"""
        try:
            logger.info("Cleaning up temporary files...")
            shutil.rmtree(self.temp_dir)
            logger.info("Cleanup complete")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Viren Portable Packager")
    parser.add_argument("--source", help="Source directory (default: current directory)")
    parser.add_argument("--output", help="Output directory (default: ./portable_build)")
    parser.add_argument("--zip", action="store_true", help="Create ZIP archive")
    parser.add_argument("--no-cleanup", action="store_true", help="Skip cleanup of temporary files")
    
    args = parser.parse_args()
    
    # Create packager
    packager = VirenPortablePackager(args.source, args.output)
    
    # Build portable version
    if packager.build_portable():
        print("Portable version built successfully!")
        
        # Create ZIP archive if requested
        if args.zip:
            zip_path = packager.create_zip_archive()
            if zip_path:
                print(f"ZIP archive created: {zip_path}")
        
        # Clean up temporary files if not skipped
        if not args.no_cleanup:
            packager.cleanup()
    else:
        print("Failed to build portable version")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())