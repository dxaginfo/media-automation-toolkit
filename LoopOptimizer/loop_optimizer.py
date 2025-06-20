#!/usr/bin/env python3
"""
LoopOptimizer - A tool for optimizing video loops for seamless playback

This tool processes video clips to create seamlessly looping versions that
are ideal for background elements, transitions, and continuous displays.
"""

import argparse
import cv2
import json
import logging
import numpy as np
import os
import tempfile
from typing import Dict, List, Optional, Tuple, Union
from google.cloud import storage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LoopOptimizer:
    """Main class for optimizing video loops."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the LoopOptimizer.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config = self._load_config(config_path)
        self._init_gcp_storage()
        logger.info("LoopOptimizer initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dict containing configuration
        """
        default_config = {
            "output_format": "mp4",
            "transition_frames": 30,
            "frame_blend_method": "linear",
            "auto_adjust_speed": True,
            "loop_detection_threshold": 0.85,
            "quality_preset": "medium",
            "use_cloud_storage": False,
            "gcp_bucket_name": "loop-optimizer-outputs"
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    return {**default_config, **user_config}
            except Exception as e:
                logger.error(f"Error loading config: {e}")
                return default_config
        return default_config
    
    def _init_gcp_storage(self):
        """Initialize Google Cloud Storage client if enabled in config."""
        if self.config.get("use_cloud_storage", False):
            try:
                self.storage_client = storage.Client()
                self.bucket = self.storage_client.bucket(self.config["gcp_bucket_name"])
                logger.info(f"GCP Storage initialized with bucket: {self.config['gcp_bucket_name']}")
            except Exception as e:
                logger.error(f"GCP Storage initialization failed: {e}")
                self.storage_client = None
                self.bucket = None
        else:
            self.storage_client = None
            self.bucket = None
    
    def optimize_loop(self, video_path: str, output_path: Optional[str] = None) -> str:
        """Optimize a video for seamless looping.
        
        Args:
            video_path: Path to input video file
            output_path: Path for output video file (optional)
            
        Returns:
            Path to optimized video file
        """
        logger.info(f"Optimizing loop for video: {video_path}")
        
        # Generate output path if not provided
        if not output_path:
            base_name = os.path.basename(video_path)
            name, ext = os.path.splitext(base_name)
            output_path = os.path.join(
                os.path.dirname(video_path),
                f"{name}_looped.{self.config['output_format']}"
            )
        
        # Extract video properties and frames
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Read all frames
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        logger.info(f"Loaded {len(frames)} frames at {fps} FPS")
        
        if len(frames) < 60:
            logger.warning("Video too short for optimal looping")
        
        # Find best loop points
        start_idx, end_idx = self._find_loop_points(frames)
        logger.info(f"Identified loop points: frames {start_idx} to {end_idx}")
        
        # Create transition between end and start
        loop_frames = self._create_seamless_loop(frames, start_idx, end_idx)
        
        # Write output video
        self._write_video(loop_frames, output_path, fps, (width, height))
        
        # Upload to cloud storage if configured
        if self.storage_client and self.bucket:
            cloud_path = self._upload_to_cloud(output_path)
            logger.info(f"Uploaded to cloud storage: {cloud_path}")
            return cloud_path
        
        return output_path
    
    def _find_loop_points(self, frames: List[np.ndarray]) -> Tuple[int, int]:
        """Find optimal start and end points for looping.
        
        Args:
            frames: List of video frames
            
        Returns:
            Tuple of (start_frame_index, end_frame_index)
        """
        # Simple implementation using frame similarity
        # In a real implementation, this would use more sophisticated algorithms
        
        # Avoid first and last 10% of frames
        min_idx = int(len(frames) * 0.1)
        max_idx = int(len(frames) * 0.9)
        
        best_similarity = 0
        best_start = min_idx
        best_end = max_idx
        
        # Sample potential loop points
        for start_idx in range(min_idx, min_idx + 30, 3):
            for end_idx in range(max_idx - 30, max_idx, 3):
                if end_idx - start_idx < 30:  # Ensure minimum loop length
                    continue
                
                # Calculate similarity between potential loop points
                similarity = self._calculate_frame_similarity(frames[start_idx], frames[end_idx])
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_start = start_idx
                    best_end = end_idx
        
        logger.info(f"Best loop point similarity: {best_similarity:.4f}")
        
        # If no good match found, use full video
        if best_similarity < self.config["loop_detection_threshold"]:
            logger.warning("No good loop points found, using full video")
            return 0, len(frames) - 1
        
        return best_start, best_end
    
    def _calculate_frame_similarity(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate similarity between two frames.
        
        Args:
            frame1: First frame
            frame2: Second frame
            
        Returns:
            Similarity score (0-1)
        """
        # Convert to grayscale for comparison
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate structural similarity index
        try:
            import skimage.metrics
            similarity = skimage.metrics.structural_similarity(gray1, gray2)
        except ImportError:
            # Fallback to simple histogram comparison
            hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
            cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
            similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        return similarity
    
    def _create_seamless_loop(self, 
                             frames: List[np.ndarray], 
                             start_idx: int, 
                             end_idx: int) -> List[np.ndarray]:
        """Create a seamless loop from identified loop points.
        
        Args:
            frames: List of all video frames
            start_idx: Starting frame index
            end_idx: Ending frame index
            
        Returns:
            List of frames for the seamless loop
        """
        # Extract frames for the main loop
        loop_frames = frames[start_idx:end_idx+1]
        
        # Create transition frames
        transition_length = min(self.config["transition_frames"], len(loop_frames) // 4)
        
        # Create blended transition between end and start
        for i in range(transition_length):
            # Calculate blend factor (0 to 1)
            alpha = i / transition_length
            
            # Get frames to blend
            end_frame = frames[end_idx - transition_length + i]
            start_frame = frames[start_idx + i]
            
            # Blend frames
            if self.config["frame_blend_method"] == "linear":
                blended = cv2.addWeighted(end_frame, 1-alpha, start_frame, alpha, 0)
            else:
                # Alternate blend method using fade through black
                blended = cv2.addWeighted(end_frame, 1-alpha, np.zeros_like(end_frame), alpha, 0)
                blended = cv2.addWeighted(blended, 1-alpha, start_frame, alpha, 0)
            
            # Replace original frame with blended version
            loop_frames[-(transition_length-i)] = blended
        
        return loop_frames
    
    def _write_video(self, 
                    frames: List[np.ndarray], 
                    output_path: str, 
                    fps: float, 
                    dimensions: Tuple[int, int]) -> None:
        """Write frames to video file.
        
        Args:
            frames: List of video frames
            output_path: Path for output video
            fps: Frames per second
            dimensions: (width, height) of the video
        """
        # Set codec and quality
        width, height = dimensions
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Or 'avc1' for H.264
        
        # Quality presets mapping
        quality_values = {
            "low": 15,
            "medium": 23,
            "high": 18,
            "ultra": 12
        }
        quality = quality_values.get(self.config["quality_preset"], 23)
        
        # Create video writer
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), True)
        
        # Write frames
        for frame in frames:
            out.write(frame)
        
        out.release()
        logger.info(f"Wrote {len(frames)} frames to {output_path}")
        
        # Apply additional encoding optimization with ffmpeg if available
        try:
            temp_path = output_path + ".temp.mp4"
            os.rename(output_path, temp_path)
            
            import subprocess
            cmd = [
                "ffmpeg", "-i", temp_path, 
                "-c:v", "libx264", "-crf", str(quality),
                "-preset", "medium", "-tune", "animation",
                "-y", output_path
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            os.remove(temp_path)
            logger.info("Applied ffmpeg optimization")
        except Exception as e:
            logger.warning(f"Failed to apply ffmpeg optimization: {e}")
            # Restore original file if ffmpeg failed
            if os.path.exists(temp_path):
                os.rename(temp_path, output_path)
    
    def _upload_to_cloud(self, file_path: str) -> str:
        """Upload file to Google Cloud Storage.
        
        Args:
            file_path: Local path to file
            
        Returns:
            Cloud storage path
        """
        if not self.bucket:
            raise ValueError("Cloud storage not initialized")
        
        file_name = os.path.basename(file_path)
        blob = self.bucket.blob(file_name)
        blob.upload_from_filename(file_path)
        
        return f"gs://{self.config['gcp_bucket_name']}/{file_name}"


def main():
    """Command line interface for LoopOptimizer."""
    parser = argparse.ArgumentParser(description="Optimize video loops for seamless playback")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--output", help="Path to output video file")
    parser.add_argument("--config", help="Path to configuration file")
    
    args = parser.parse_args()
    
    optimizer = LoopOptimizer(args.config)
    output_path = optimizer.optimize_loop(args.video, args.output)
    
    print(f"Optimized loop saved to: {output_path}")


if __name__ == "__main__":
    main()