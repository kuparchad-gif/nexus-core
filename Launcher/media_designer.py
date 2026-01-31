"""
Media Designer Module
Complete toolkit for creating, editing, and processing images, videos, and audio
"""

from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class MediaDesigner:
    """
    Comprehensive media design capabilities:
    - Image creation and editing
    - Image filters and effects
    - Video processing and editing
    - Frame extraction and manipulation
    - Format conversion
    - Batch processing
    """
    
    def __init__(self, workspace: str = "/tmp/media"):
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        
    # ==================== IMAGE CREATION ====================
    
    def create_image(self, width: int, height: int, color: str, 
                     output_filename: str, format: str = "PNG") -> str:
        """Create a solid color image"""
        try:
            img = Image.new('RGB', (width, height), color)
            output_path = self.workspace / output_filename
            img.save(output_path, format=format)
            logger.info(f"Image created: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Image creation failed: {e}")
            raise
    
    def create_gradient(self, width: int, height: int, 
                        color1: Tuple[int, int, int], 
                        color2: Tuple[int, int, int],
                        direction: str, output_filename: str) -> str:
        """
        Create gradient image
        direction: 'horizontal', 'vertical', 'diagonal'
        """
        try:
            img = Image.new('RGB', (width, height))
            draw = ImageDraw.Draw(img)
            
            if direction == 'horizontal':
                for x in range(width):
                    ratio = x / width
                    r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
                    g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
                    b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
                    draw.line([(x, 0), (x, height)], fill=(r, g, b))
            
            elif direction == 'vertical':
                for y in range(height):
                    ratio = y / height
                    r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
                    g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
                    b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
                    draw.line([(0, y), (width, y)], fill=(r, g, b))
            
            elif direction == 'diagonal':
                for y in range(height):
                    for x in range(width):
                        ratio = (x + y) / (width + height)
                        r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
                        g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
                        b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
                        draw.point((x, y), fill=(r, g, b))
            
            output_path = self.workspace / output_filename
            img.save(output_path)
            logger.info(f"Gradient created: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Gradient creation failed: {e}")
            raise
    
    def add_text_to_image(self, image_path: str, text: str, 
                          position: Tuple[int, int], 
                          font_size: int = 40,
                          color: str = "white",
                          output_filename: Optional[str] = None) -> str:
        """Add text overlay to image"""
        try:
            img = Image.open(image_path)
            draw = ImageDraw.Draw(img)
            
            # Use default font (can be enhanced with custom fonts)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            except:
                font = ImageFont.load_default()
            
            draw.text(position, text, font=font, fill=color)
            
            if output_filename is None:
                output_filename = f"text_{Path(image_path).name}"
            
            output_path = self.workspace / output_filename
            img.save(output_path)
            logger.info(f"Text added to image: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Adding text failed: {e}")
            raise
    
    def create_collage(self, image_paths: List[str], 
                       layout: Tuple[int, int],
                       output_filename: str,
                       spacing: int = 10) -> str:
        """
        Create image collage
        layout: (rows, cols)
        """
        try:
            images = [Image.open(path) for path in image_paths]
            
            # Resize all images to same size
            target_size = images[0].size
            images = [img.resize(target_size) for img in images]
            
            rows, cols = layout
            img_width, img_height = target_size
            
            # Calculate collage dimensions
            collage_width = cols * img_width + (cols + 1) * spacing
            collage_height = rows * img_height + (rows + 1) * spacing
            
            collage = Image.new('RGB', (collage_width, collage_height), 'white')
            
            for idx, img in enumerate(images[:rows * cols]):
                row = idx // cols
                col = idx % cols
                x = col * img_width + (col + 1) * spacing
                y = row * img_height + (row + 1) * spacing
                collage.paste(img, (x, y))
            
            output_path = self.workspace / output_filename
            collage.save(output_path)
            logger.info(f"Collage created: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Collage creation failed: {e}")
            raise
    
    # ==================== IMAGE EDITING ====================
    
    def resize_image(self, image_path: str, width: int, height: int,
                     output_filename: Optional[str] = None) -> str:
        """Resize image to specified dimensions"""
        try:
            img = Image.open(image_path)
            img_resized = img.resize((width, height), Image.Resampling.LANCZOS)
            
            if output_filename is None:
                output_filename = f"resized_{Path(image_path).name}"
            
            output_path = self.workspace / output_filename
            img_resized.save(output_path)
            logger.info(f"Image resized: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Resize failed: {e}")
            raise
    
    def crop_image(self, image_path: str, box: Tuple[int, int, int, int],
                   output_filename: Optional[str] = None) -> str:
        """
        Crop image
        box: (left, top, right, bottom)
        """
        try:
            img = Image.open(image_path)
            img_cropped = img.crop(box)
            
            if output_filename is None:
                output_filename = f"cropped_{Path(image_path).name}"
            
            output_path = self.workspace / output_filename
            img_cropped.save(output_path)
            logger.info(f"Image cropped: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Crop failed: {e}")
            raise
    
    def rotate_image(self, image_path: str, angle: float,
                     expand: bool = True,
                     output_filename: Optional[str] = None) -> str:
        """Rotate image by angle (degrees)"""
        try:
            img = Image.open(image_path)
            img_rotated = img.rotate(angle, expand=expand)
            
            if output_filename is None:
                output_filename = f"rotated_{Path(image_path).name}"
            
            output_path = self.workspace / output_filename
            img_rotated.save(output_path)
            logger.info(f"Image rotated: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Rotation failed: {e}")
            raise
    
    def flip_image(self, image_path: str, direction: str,
                   output_filename: Optional[str] = None) -> str:
        """
        Flip image
        direction: 'horizontal', 'vertical'
        """
        try:
            img = Image.open(image_path)
            
            if direction == 'horizontal':
                img_flipped = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            elif direction == 'vertical':
                img_flipped = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            else:
                raise ValueError(f"Unknown direction: {direction}")
            
            if output_filename is None:
                output_filename = f"flipped_{Path(image_path).name}"
            
            output_path = self.workspace / output_filename
            img_flipped.save(output_path)
            logger.info(f"Image flipped: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Flip failed: {e}")
            raise
    
    # ==================== IMAGE FILTERS & EFFECTS ====================
    
    def apply_filter(self, image_path: str, filter_name: str,
                     output_filename: Optional[str] = None) -> str:
        """
        Apply filter to image
        filters: blur, sharpen, edge_enhance, emboss, contour, smooth
        """
        try:
            img = Image.open(image_path)
            
            filters = {
                'blur': ImageFilter.BLUR,
                'sharpen': ImageFilter.SHARPEN,
                'edge_enhance': ImageFilter.EDGE_ENHANCE,
                'emboss': ImageFilter.EMBOSS,
                'contour': ImageFilter.CONTOUR,
                'smooth': ImageFilter.SMOOTH
            }
            
            if filter_name not in filters:
                raise ValueError(f"Unknown filter: {filter_name}")
            
            img_filtered = img.filter(filters[filter_name])
            
            if output_filename is None:
                output_filename = f"{filter_name}_{Path(image_path).name}"
            
            output_path = self.workspace / output_filename
            img_filtered.save(output_path)
            logger.info(f"Filter applied: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Filter application failed: {e}")
            raise
    
    def adjust_brightness(self, image_path: str, factor: float,
                          output_filename: Optional[str] = None) -> str:
        """Adjust image brightness (factor: 0.0 to 2.0)"""
        try:
            img = Image.open(image_path)
            enhancer = ImageEnhance.Brightness(img)
            img_enhanced = enhancer.enhance(factor)
            
            if output_filename is None:
                output_filename = f"brightness_{Path(image_path).name}"
            
            output_path = self.workspace / output_filename
            img_enhanced.save(output_path)
            logger.info(f"Brightness adjusted: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Brightness adjustment failed: {e}")
            raise
    
    def adjust_contrast(self, image_path: str, factor: float,
                        output_filename: Optional[str] = None) -> str:
        """Adjust image contrast (factor: 0.0 to 2.0)"""
        try:
            img = Image.open(image_path)
            enhancer = ImageEnhance.Contrast(img)
            img_enhanced = enhancer.enhance(factor)
            
            if output_filename is None:
                output_filename = f"contrast_{Path(image_path).name}"
            
            output_path = self.workspace / output_filename
            img_enhanced.save(output_path)
            logger.info(f"Contrast adjusted: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Contrast adjustment failed: {e}")
            raise
    
    def convert_to_grayscale(self, image_path: str,
                             output_filename: Optional[str] = None) -> str:
        """Convert image to grayscale"""
        try:
            img = Image.open(image_path)
            img_gray = img.convert('L')
            
            if output_filename is None:
                output_filename = f"gray_{Path(image_path).name}"
            
            output_path = self.workspace / output_filename
            img_gray.save(output_path)
            logger.info(f"Converted to grayscale: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Grayscale conversion failed: {e}")
            raise
    
    # ==================== VIDEO PROCESSING ====================
    
    def extract_frames(self, video_path: str, output_dir: Optional[str] = None,
                       frame_interval: int = 30) -> List[str]:
        """
        Extract frames from video
        frame_interval: extract every Nth frame
        """
        try:
            if output_dir is None:
                output_dir = self.workspace / f"frames_{Path(video_path).stem}"
            else:
                output_dir = Path(output_dir)
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            saved_frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    frame_filename = output_dir / f"frame_{frame_count:06d}.jpg"
                    cv2.imwrite(str(frame_filename), frame)
                    saved_frames.append(str(frame_filename))
                
                frame_count += 1
            
            cap.release()
            logger.info(f"Extracted {len(saved_frames)} frames to {output_dir}")
            return saved_frames
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            raise
    
    def create_video_from_images(self, image_paths: List[str], 
                                 output_filename: str,
                                 fps: int = 30) -> str:
        """Create video from sequence of images"""
        try:
            if not image_paths:
                raise ValueError("No images provided")
            
            # Read first image to get dimensions
            first_frame = cv2.imread(image_paths[0])
            height, width, _ = first_frame.shape
            
            output_path = self.workspace / output_filename
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            for image_path in image_paths:
                frame = cv2.imread(image_path)
                if frame is not None:
                    # Resize if dimensions don't match
                    if frame.shape[:2] != (height, width):
                        frame = cv2.resize(frame, (width, height))
                    video.write(frame)
            
            video.release()
            logger.info(f"Video created: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Video creation failed: {e}")
            raise
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get video metadata"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            info = {
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "duration_seconds": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
            }
            
            cap.release()
            return info
        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
            raise
    
    def resize_video(self, video_path: str, width: int, height: int,
                     output_filename: str) -> str:
        """Resize video to specified dimensions"""
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            output_path = self.workspace / output_filename
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                resized_frame = cv2.resize(frame, (width, height))
                video.write(resized_frame)
            
            cap.release()
            video.release()
            logger.info(f"Video resized: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Video resize failed: {e}")
            raise
    
    # ==================== FORMAT CONVERSION ====================
    
    def convert_image_format(self, image_path: str, target_format: str,
                             output_filename: Optional[str] = None) -> str:
        """Convert image to different format (PNG, JPEG, BMP, etc.)"""
        try:
            img = Image.open(image_path)
            
            if output_filename is None:
                output_filename = f"{Path(image_path).stem}.{target_format.lower()}"
            
            output_path = self.workspace / output_filename
            
            # Handle RGBA to RGB conversion for JPEG
            if target_format.upper() == 'JPEG' and img.mode == 'RGBA':
                img = img.convert('RGB')
            
            img.save(output_path, format=target_format.upper())
            logger.info(f"Image converted: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Format conversion failed: {e}")
            raise
    
    # ==================== BATCH PROCESSING ====================
    
    def batch_resize(self, image_paths: List[str], width: int, height: int) -> List[str]:
        """Batch resize multiple images"""
        output_paths = []
        for image_path in image_paths:
            try:
                output_path = self.resize_image(image_path, width, height)
                output_paths.append(output_path)
            except Exception as e:
                logger.error(f"Failed to resize {image_path}: {e}")
        
        return output_paths
    
    def batch_apply_filter(self, image_paths: List[str], filter_name: str) -> List[str]:
        """Batch apply filter to multiple images"""
        output_paths = []
        for image_path in image_paths:
            try:
                output_path = self.apply_filter(image_path, filter_name)
                output_paths.append(output_path)
            except Exception as e:
                logger.error(f"Failed to filter {image_path}: {e}")
        
        return output_paths
    
    def batch_convert_format(self, image_paths: List[str], target_format: str) -> List[str]:
        """Batch convert images to target format"""
        output_paths = []
        for image_path in image_paths:
            try:
                output_path = self.convert_image_format(image_path, target_format)
                output_paths.append(output_path)
            except Exception as e:
                logger.error(f"Failed to convert {image_path}: {e}")
        
        return output_paths
