#!/usr/bin/env python3
"""
Viren Document Suite - Comprehensive document processing interface
"""

import os
import json
import time
import logging
import gradio as gr
import tempfile
import shutil
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logger = logging.getLogger("VirenDocumentSuite")

class VirenDocumentSuite:
    """
    Comprehensive document processing suite for Viren Platinum Edition
    """
    
    def __init__(self, upload_dir: str = "uploads", output_dir: str = "documents"):
        """Initialize the document suite"""
        self.upload_dir = upload_dir
        self.output_dir = output_dir
        self.temp_dir = tempfile.mkdtemp()
        
        # Create directories if they don't exist
        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize document processor
        try:
            from document_processor import DocumentProcessor
            self.doc_processor = DocumentProcessor(upload_dir, os.path.join(output_dir, "cache"))
            logger.info("Document processor initialized")
        except ImportError:
            logger.warning("document_processor module not found, using placeholder")
            self.doc_processor = None
        
        # Supported file types
        self.supported_types = {
            "document": [".pdf", ".docx", ".doc", ".rtf", ".txt", ".md"],
            "spreadsheet": [".xlsx", ".xls", ".csv", ".tsv"],
            "presentation": [".pptx", ".ppt"],
            "image": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"],
            "diagram": [".vsdx", ".vsd", ".drawio"],
            "code": [".py", ".js", ".html", ".css", ".java", ".cpp", ".c", ".h", ".json", ".xml"]
        }
    
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface for the document suite"""
        with gr.Blocks(title="Viren Document Suite") as interface:
            gr.Markdown("# Viren Document Suite")
            gr.Markdown("Comprehensive document processing and management")
            
            with gr.Tabs() as tabs:
                # Document Viewer Tab
                with gr.TabItem("📄 Document Viewer"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            file_upload = gr.File(
                                label="Upload Document",
                                file_types=self._get_all_extensions(),
                                type="filepath"
                            )
                            
                            file_info = gr.JSON(
                                label="Document Information",
                                value={}
                            )
                            
                            view_btn = gr.Button("View Document", variant="primary")
                        
                        with gr.Column(scale=2):
                            document_view = gr.HTML(
                                label="Document Preview",
                                value="<div class='doc-placeholder'>Upload a document to view its contents</div>"
                            )
                
                # Document Editor Tab
                with gr.TabItem("✏️ Document Editor"):
                    with gr.Row():
                        with gr.Column():
                            editor_file = gr.File(
                                label="Open Document",
                                file_types=[".txt", ".md", ".html", ".css", ".js", ".py", ".json", ".xml", ".csv"],
                                type="filepath"
                            )
                            
                            editor_content = gr.Textbox(
                                label="Document Content",
                                lines=20,
                                max_lines=50
                            )
                            
                            with gr.Row():
                                save_btn = gr.Button("Save Changes", variant="primary")
                                download_btn = gr.Button("Download")
                
                # Spreadsheet Viewer Tab
                with gr.TabItem("📊 Spreadsheet Viewer"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            sheet_upload = gr.File(
                                label="Upload Spreadsheet",
                                file_types=[".xlsx", ".xls", ".csv", ".tsv"],
                                type="filepath"
                            )
                            
                            sheet_info = gr.JSON(
                                label="Spreadsheet Information",
                                value={}
                            )
                            
                            sheet_selector = gr.Dropdown(
                                label="Select Sheet",
                                choices=[],
                                interactive=True
                            )
                        
                        with gr.Column(scale=2):
                            sheet_view = gr.Dataframe(
                                label="Spreadsheet Data",
                                interactive=False
                            )
                
                # Image Viewer Tab
                with gr.TabItem("🖼️ Image Viewer"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            image_upload = gr.File(
                                label="Upload Image",
                                file_types=[".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"],
                                type="filepath"
                            )
                            
                            image_info = gr.JSON(
                                label="Image Information",
                                value={}
                            )
                        
                        with gr.Column(scale=2):
                            image_view = gr.Image(
                                label="Image Preview",
                                type="filepath"
                            )
                
                # Code Editor Tab
                with gr.TabItem("💻 Code Editor"):
                    with gr.Row():
                        with gr.Column():
                            code_file = gr.File(
                                label="Open Code File",
                                file_types=[".py", ".js", ".html", ".css", ".java", ".cpp", ".c", ".h", ".json", ".xml"],
                                type="filepath"
                            )
                            
                            language_selector = gr.Dropdown(
                                label="Language",
                                choices=["python", "javascript", "html", "css", "java", "cpp", "json", "xml"],
                                value="python",
                                interactive=True
                            )
                            
                            code_editor = gr.Code(
                                label="Code Editor",
                                language="python",
                                lines=20,
                                interactive=True
                            )
                            
                            with gr.Row():
                                save_code_btn = gr.Button("Save Code", variant="primary")
                                run_code_btn = gr.Button("Run Code")
                                format_code_btn = gr.Button("Format Code")
                        
                        with gr.Column():
                            code_output = gr.Textbox(
                                label="Output",
                                lines=10,
                                max_lines=20
                            )
                
                # Diagram Editor Tab
                with gr.TabItem("📐 Diagram Editor"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### Diagram Tools")
                            
                            diagram_tool = gr.Radio(
                                ["Select", "Draw", "Text", "Shape", "Connector"],
                                label="Tool",
                                value="Select"
                            )
                            
                            diagram_color = gr.ColorPicker(
                                label="Color",
                                value="#3b82f6"
                            )
                            
                            with gr.Row():
                                save_diagram_btn = gr.Button("Save Diagram", variant="primary")
                                export_diagram_btn = gr.Button("Export Diagram")
                        
                        with gr.Column(scale=3):
                            diagram_canvas = gr.HTML(
                                value="<div id='diagram-canvas' class='diagram-canvas'>Canvas loading...</div>"
                            )
                
                # Batch Processing Tab
                with gr.TabItem("🔄 Batch Processing"):
                    with gr.Row():
                        with gr.Column():
                            batch_files = gr.File(
                                label="Upload Files",
                                file_types=self._get_all_extensions(),
                                file_count="multiple",
                                type="filepath"
                            )
                            
                            process_type = gr.Dropdown(
                                label="Process Type",
                                choices=["Extract Text", "Convert Format", "Analyze Content", "Generate Summary"],
                                value="Extract Text",
                                interactive=True
                            )
                            
                            batch_process_btn = gr.Button("Process Files", variant="primary")
                        
                        with gr.Column():
                            batch_results = gr.Dataframe(
                                label="Processing Results",
                                headers=["Filename", "Type", "Status", "Output"]
                            )
                            
                            batch_download_btn = gr.Button("Download Results")
            
            # Event handlers
            def view_document(file_path):
                if not file_path:
                    return {}, "<div class='doc-placeholder'>No document selected</div>"
                
                try:
                    # Process document
                    if self.doc_processor:
                        result = self.doc_processor.process_document(file_path)
                        html_preview = self.doc_processor.generate_html_preview(result)
                        return result, html_preview
                    else:
                        # Placeholder implementation
                        file_name = os.path.basename(file_path)
                        file_ext = os.path.splitext(file_name)[1].lower()
                        file_size = os.path.getsize(file_path) / 1024  # KB
                        
                        info = {
                            "filename": file_name,
                            "type": file_ext,
                            "size": f"{file_size:.2f} KB",
                            "path": file_path
                        }
                        
                        html = f"<div class='doc-preview'><h3>{file_name}</h3><p>File size: {file_size:.2f} KB</p></div>"
                        return info, html
                except Exception as e:
                    logger.error(f"Error viewing document: {e}")
                    return {"error": str(e)}, f"<div class='error'>Error: {str(e)}</div>"
            
            view_btn.click(
                fn=view_document,
                inputs=[file_upload],
                outputs=[file_info, document_view]
            )
            
            def load_editor_file(file_path):
                if not file_path:
                    return ""
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    return content
                except UnicodeDecodeError:
                    try:
                        with open(file_path, 'r', encoding='latin-1') as f:
                            content = f.read()
                        return content
                    except Exception as e:
                        logger.error(f"Error loading file: {e}")
                        return f"Error loading file: {str(e)}"
                except Exception as e:
                    logger.error(f"Error loading file: {e}")
                    return f"Error loading file: {str(e)}"
            
            editor_file.change(
                fn=load_editor_file,
                inputs=[editor_file],
                outputs=[editor_content]
            )
            
            def save_editor_content(file_path, content):
                if not file_path:
                    return "No file selected"
                
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    return f"Saved changes to {os.path.basename(file_path)}"
                except Exception as e:
                    logger.error(f"Error saving file: {e}")
                    return f"Error saving file: {str(e)}"
            
            save_btn.click(
                fn=save_editor_content,
                inputs=[editor_file, editor_content],
                outputs=[gr.Textbox(visible=True)]
            )
            
            def load_spreadsheet(file_path):
                if not file_path:
                    return {}, [], None
                
                try:
                    import pandas as pd
                    
                    file_name = os.path.basename(file_path)
                    file_ext = os.path.splitext(file_name)[1].lower()
                    
                    if file_ext == ".csv":
                        df = pd.read_csv(file_path)
                        sheet_names = ["Sheet1"]
                        info = {
                            "filename": file_name,
                            "type": "CSV",
                            "rows": len(df),
                            "columns": len(df.columns)
                        }
                        return info, sheet_names, df
                    elif file_ext in [".xlsx", ".xls"]:
                        xls = pd.ExcelFile(file_path)
                        sheet_names = xls.sheet_names
                        df = pd.read_excel(file_path, sheet_name=sheet_names[0])
                        info = {
                            "filename": file_name,
                            "type": "Excel",
                            "sheets": sheet_names,
                            "rows": len(df),
                            "columns": len(df.columns)
                        }
                        return info, sheet_names, df
                    else:
                        return {"error": "Unsupported file type"}, [], None
                except Exception as e:
                    logger.error(f"Error loading spreadsheet: {e}")
                    return {"error": str(e)}, [], None
            
            sheet_upload.change(
                fn=load_spreadsheet,
                inputs=[sheet_upload],
                outputs=[sheet_info, sheet_selector, sheet_view]
            )
            
            def change_sheet(file_path, sheet_name):
                if not file_path or not sheet_name:
                    return None
                
                try:
                    import pandas as pd
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    return df
                except Exception as e:
                    logger.error(f"Error changing sheet: {e}")
                    return None
            
            sheet_selector.change(
                fn=change_sheet,
                inputs=[sheet_upload, sheet_selector],
                outputs=[sheet_view]
            )
            
            def load_image(file_path):
                if not file_path:
                    return {}, None
                
                try:
                    from PIL import Image
                    
                    img = Image.open(file_path)
                    file_name = os.path.basename(file_path)
                    file_size = os.path.getsize(file_path) / 1024  # KB
                    
                    info = {
                        "filename": file_name,
                        "type": img.format,
                        "size": f"{file_size:.2f} KB",
                        "dimensions": f"{img.width} x {img.height}",
                        "mode": img.mode
                    }
                    
                    return info, file_path
                except Exception as e:
                    logger.error(f"Error loading image: {e}")
                    return {"error": str(e)}, None
            
            image_upload.change(
                fn=load_image,
                inputs=[image_upload],
                outputs=[image_info, image_view]
            )
            
            def load_code_file(file_path):
                if not file_path:
                    return "", "python"
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Determine language from file extension
                    file_ext = os.path.splitext(file_path)[1].lower()
                    language = "python"  # Default
                    
                    if file_ext == ".py":
                        language = "python"
                    elif file_ext == ".js":
                        language = "javascript"
                    elif file_ext == ".html":
                        language = "html"
                    elif file_ext == ".css":
                        language = "css"
                    elif file_ext in [".java"]:
                        language = "java"
                    elif file_ext in [".cpp", ".c", ".h"]:
                        language = "cpp"
                    elif file_ext == ".json":
                        language = "json"
                    elif file_ext == ".xml":
                        language = "xml"
                    
                    return content, language
                except Exception as e:
                    logger.error(f"Error loading code file: {e}")
                    return f"# Error loading file: {str(e)}", "python"
            
            code_file.change(
                fn=load_code_file,
                inputs=[code_file],
                outputs=[code_editor, language_selector]
            )
            
            def update_code_language(language):
                return gr.Code.update(language=language)
            
            language_selector.change(
                fn=update_code_language,
                inputs=[language_selector],
                outputs=[code_editor]
            )
            
            def save_code(file_path, code):
                if not file_path:
                    return "No file selected"
                
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(code)
                    return f"Saved code to {os.path.basename(file_path)}"
                except Exception as e:
                    logger.error(f"Error saving code: {e}")
                    return f"Error saving code: {str(e)}"
            
            save_code_btn.click(
                fn=save_code,
                inputs=[code_file, code_editor],
                outputs=[code_output]
            )
            
            def run_code(code, language):
                if language != "python":
                    return "Only Python code execution is supported"
                
                try:
                    # Create a temporary file
                    temp_file = os.path.join(self.temp_dir, "temp_code.py")
                    with open(temp_file, 'w', encoding='utf-8') as f:
                        f.write(code)
                    
                    # Run the code and capture output
                    import subprocess
                    result = subprocess.run(
                        [sys.executable, temp_file],
                        capture_output=True,
                        text=True,
                        timeout=10  # 10 second timeout
                    )
                    
                    # Combine stdout and stderr
                    output = result.stdout
                    if result.stderr:
                        output += "\n" + result.stderr
                    
                    return output
                except subprocess.TimeoutExpired:
                    return "Execution timed out (10 second limit)"
                except Exception as e:
                    logger.error(f"Error running code: {e}")
                    return f"Error running code: {str(e)}"
            
            run_code_btn.click(
                fn=run_code,
                inputs=[code_editor, language_selector],
                outputs=[code_output]
            )
            
            def format_code(code, language):
                if language != "python":
                    return code, "Only Python code formatting is supported"
                
                try:
                    import autopep8
                    formatted_code = autopep8.fix_code(code)
                    return formatted_code, "Code formatted successfully"
                except ImportError:
                    return code, "autopep8 not installed, formatting unavailable"
                except Exception as e:
                    logger.error(f"Error formatting code: {e}")
                    return code, f"Error formatting code: {str(e)}"
            
            format_code_btn.click(
                fn=format_code,
                inputs=[code_editor, language_selector],
                outputs=[code_editor, code_output]
            )
            
            def process_batch_files(files, process_type):
                if not files:
                    return []
                
                results = []
                for file_path in files:
                    try:
                        file_name = os.path.basename(file_path)
                        file_ext = os.path.splitext(file_name)[1].lower()
                        file_type = "Unknown"
                        
                        for type_name, extensions in self.supported_types.items():
                            if file_ext in extensions:
                                file_type = type_name
                                break
                        
                        # Process based on type
                        if process_type == "Extract Text":
                            output = self._extract_text(file_path, file_type)
                            status = "Success" if output else "Failed"
                        elif process_type == "Convert Format":
                            output = "Conversion not implemented"
                            status = "Not Implemented"
                        elif process_type == "Analyze Content":
                            output = "Analysis not implemented"
                            status = "Not Implemented"
                        elif process_type == "Generate Summary":
                            output = "Summary not implemented"
                            status = "Not Implemented"
                        else:
                            output = "Unknown process type"
                            status = "Failed"
                        
                        results.append([file_name, file_type, status, output])
                    except Exception as e:
                        logger.error(f"Error processing file {file_path}: {e}")
                        results.append([os.path.basename(file_path), "Unknown", "Error", str(e)])
                
                return results
            
            batch_process_btn.click(
                fn=process_batch_files,
                inputs=[batch_files, process_type],
                outputs=[batch_results]
            )
        
        return interface
    
    def _extract_text(self, file_path: str, file_type: str) -> str:
        """Extract text from a file"""
        if self.doc_processor:
            result = self.doc_processor.process_document(file_path)
            if "preview" in result:
                return result["preview"]
        
        # Fallback implementation
        if file_type == "document":
            try:
                if file_path.endswith(".pdf"):
                    import PyPDF2
                    with open(file_path, 'rb') as f:
                        reader = PyPDF2.PdfReader(f)
                        text = ""
                        for page in reader.pages:
                            text += page.extract_text() + "\n"
                        return text
                elif file_path.endswith(".docx"):
                    import docx
                    doc = docx.Document(file_path)
                    return "\n".join([para.text for para in doc.paragraphs])
                elif file_path.endswith((".txt", ".md")):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return f.read()
            except Exception as e:
                logger.error(f"Error extracting text: {e}")
                return f"Error: {str(e)}"
        
        return "Text extraction not supported for this file type"
    
    def _get_all_extensions(self) -> List[str]:
        """Get all supported file extensions"""
        extensions = []
        for type_exts in self.supported_types.values():
            extensions.extend(type_exts)
        return extensions
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            logger.error(f"Error cleaning up temporary files: {e}")

# For importing
import sys

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create document suite
    doc_suite = VirenDocumentSuite()
    
    # Create and launch interface
    interface = doc_suite.create_interface()
    interface.launch()
    
    # Clean up on exit
    doc_suite.cleanup()