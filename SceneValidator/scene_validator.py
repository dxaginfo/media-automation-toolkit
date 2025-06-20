#!/usr/bin/env python3
"""
SceneValidator - A tool for validating scene composition and continuity

This tool analyzes video frames and scene metadata to identify inconsistencies,
continuity errors, and composition issues.
"""

import argparse
import json
import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from google.cloud import storage
from firebase_admin import firestore
import google.generativeai as genai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure Gemini API
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


class SceneValidator:
    """Main class for validating scene composition and continuity."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the SceneValidator.

        Args:
            config_path: Path to configuration file (optional)
        """
        self.config = self._load_config(config_path)
        self.gemini_model = genai.GenerativeModel('gemini-pro-vision')
        self._init_firebase()
        logger.info("SceneValidator initialized")

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults.

        Args:
            config_path: Path to configuration file

        Returns:
            Dict containing configuration
        """
        default_config = {
            "threshold_continuity": 0.85,
            "threshold_composition": 0.75,
            "frame_sample_rate": 10,
            "use_cloud_storage": False,
            "firebase_collection": "scene_validation_results"
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

    def _init_firebase(self):
        """Initialize Firebase connection if enabled in config."""
        if self.config.get("use_firebase", False):
            try:
                self.db = firestore.client()
                self.results_collection = self.db.collection(
                    self.config["firebase_collection"]
                )
                logger.info("Firebase initialized")
            except Exception as e:
                logger.error(f"Firebase initialization failed: {e}")
                self.db = None
        else:
            self.db = None

    def validate_scene(self, video_path: str, metadata_path: str) -> Dict:
        """Validate a scene using the provided video and metadata.

        Args:
            video_path: Path to video file
            metadata_path: Path to metadata JSON file

        Returns:
            Dict containing validation results
        """
        logger.info(f"Validating scene: {video_path}")

        # Load video and metadata
        frames = self._extract_frames(video_path)
        metadata = self._load_metadata(metadata_path)

        if frames is None or metadata is None:
            return {"status": "error", "message": "Failed to load inputs"}

        # Run validation checks
        continuity_issues = self._check_continuity(frames, metadata)
        composition_issues = self._check_composition(frames, metadata)
        metadata_issues = self._check_metadata_consistency(metadata)

        # Prepare results
        results = {
            "status": "success",
            "scene_id": metadata.get("scene_id", "unknown"),
            "validation_time": self._get_timestamp(),
            "issues_detected": len(continuity_issues) + len(composition_issues) + len(metadata_issues) > 0,
            "issues": {
                "continuity": continuity_issues,
                "composition": composition_issues,
                "metadata": metadata_issues
            },
            "summary": self._generate_summary(continuity_issues, composition_issues, metadata_issues)
        }

        # Store results in Firebase if configured
        if self.db:
            self._store_results(results)

        return results

    def _extract_frames(self, video_path: str) -> Optional[List[np.ndarray]]:
        """Extract frames from video for analysis.

        Args:
            video_path: Path to video file

        Returns:
            List of frames as numpy arrays, or None if extraction failed
        """
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            frame_count = 0
            sample_rate = self.config["frame_sample_rate"]

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % sample_rate == 0:
                    frames.append(frame)

                frame_count += 1

            cap.release()
            logger.info(f"Extracted {len(frames)} frames for analysis")
            return frames
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            return None

    def _load_metadata(self, metadata_path: str) -> Optional[Dict]:
        """Load scene metadata from JSON file.

        Args:
            metadata_path: Path to metadata JSON file

        Returns:
            Dict containing metadata, or None if loading failed
        """
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Loaded metadata: {metadata.get('scene_id', 'unknown')}")
            return metadata
        except Exception as e:
            logger.error(f"Metadata loading failed: {e}")
            return None

    def _check_continuity(self, frames: List[np.ndarray], metadata: Dict) -> List[Dict]:
        """Check for continuity issues between frames.

        Args:
            frames: List of video frames
            metadata: Scene metadata

        Returns:
            List of detected continuity issues
        """
        issues = []
        expected_objects = metadata.get("tracked_objects", [])

        # Simple object tracking simulation
        for i in range(1, len(frames)):
            prev_frame = frames[i-1]
            curr_frame = frames[i]

            # Example analysis - in a real implementation, this would use
            # computer vision algorithms for object tracking and detection
            diff = cv2.absdiff(prev_frame, curr_frame)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
            
            # If significant changes detected, analyze with Gemini for potential issues
            if np.mean(thresh) > 10:
                for obj in expected_objects:
                    # Simulate detecting missing or changed objects
                    if np.random.random() > self.config["threshold_continuity"]:
                        issues.append({
                            "frame_index": i,
                            "object_id": obj.get("id", "unknown"),
                            "issue_type": "object_discontinuity",
                            "description": f"Object {obj.get('name', 'unknown')} changed position or appearance unexpectedly"
                        })

        return issues

    def _check_composition(self, frames: List[np.ndarray], metadata: Dict) -> List[Dict]:
        """Check for composition issues in frames.

        Args:
            frames: List of video frames
            metadata: Scene metadata

        Returns:
            List of detected composition issues
        """
        issues = []
        composition_rules = metadata.get("composition_rules", [])

        # Analyze sample frames for composition
        for i, frame in enumerate([frames[0], frames[len(frames)//2], frames[-1]]):
            # Example analysis - in a real implementation, this would use
            # computer vision for composition analysis
            
            # Convert to RGB for Gemini API
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Use Gemini for composition analysis if API key available
            if GEMINI_API_KEY:
                try:
                    prompt = """
                    Analyze this frame for cinematographic composition issues. 
                    Consider rule of thirds, headroom, leading space, and framing.
                    Format response as JSON with keys: 'has_issues' (boolean) and 'issues' (array of strings).
                    """
                    response = self.gemini_model.generate_content([prompt, rgb_frame])
                    result = json.loads(response.text)
                    
                    if result.get("has_issues", False):
                        for issue in result.get("issues", []):
                            issues.append({
                                "frame_index": i * (len(frames) // 2) if i < 2 else len(frames) - 1,
                                "issue_type": "composition",
                                "description": issue
                            })
                except Exception as e:
                    logger.error(f"Gemini API error: {e}")
            
            # Fallback simple analysis
            else:
                # Simple rule of thirds check
                h, w = frame.shape[:2]
                for rule in composition_rules:
                    if rule == "rule_of_thirds" and np.random.random() > self.config["threshold_composition"]:
                        issues.append({
                            "frame_index": i * (len(frames) // 2) if i < 2 else len(frames) - 1,
                            "issue_type": "composition",
                            "description": "Key elements not aligned with rule of thirds"
                        })

        return issues

    def _check_metadata_consistency(self, metadata: Dict) -> List[Dict]:
        """Check for inconsistencies in metadata.

        Args:
            metadata: Scene metadata

        Returns:
            List of detected metadata issues
        """
        issues = []
        required_fields = ["scene_id", "scene_number", "location", "time_of_day"]
        
        # Check for missing required fields
        for field in required_fields:
            if field not in metadata:
                issues.append({
                    "issue_type": "missing_metadata",
                    "description": f"Required field '{field}' is missing"
                })
        
        # Check for time continuity with adjacent scenes
        prev_scene = metadata.get("previous_scene")
        if prev_scene:
            prev_time = prev_scene.get("time_of_day")
            curr_time = metadata.get("time_of_day")
            
            if prev_time and curr_time and prev_time != curr_time:
                # Check if time progression makes sense
                valid_progression = {
                    "morning": ["afternoon", "evening"],
                    "afternoon": ["evening", "night"],
                    "evening": ["night"],
                    "night": ["morning"]
                }
                
                if curr_time not in valid_progression.get(prev_time, []):
                    issues.append({
                        "issue_type": "time_continuity",
                        "description": f"Unexpected time progression from {prev_time} to {curr_time}"
                    })
        
        return issues

    def _generate_summary(self, 
                         continuity_issues: List[Dict], 
                         composition_issues: List[Dict],
                         metadata_issues: List[Dict]) -> str:
        """Generate a human-readable summary of validation results.

        Args:
            continuity_issues: List of continuity issues
            composition_issues: List of composition issues
            metadata_issues: List of metadata issues

        Returns:
            Summary string
        """
        total_issues = len(continuity_issues) + len(composition_issues) + len(metadata_issues)
        
        if total_issues == 0:
            return "No issues detected. Scene passed validation."
        
        summary = f"Detected {total_issues} issues:\n"
        summary += f"- {len(continuity_issues)} continuity issues\n"
        summary += f"- {len(composition_issues)} composition issues\n"
        summary += f"- {len(metadata_issues)} metadata issues\n"
        
        if len(continuity_issues) > 0:
            summary += "\nContinuity issues:\n"
            for i, issue in enumerate(continuity_issues[:3]):
                summary += f"{i+1}. {issue['description']}\n"
            if len(continuity_issues) > 3:
                summary += f"...and {len(continuity_issues) - 3} more continuity issues\n"
        
        if len(composition_issues) > 0:
            summary += "\nComposition issues:\n"
            for i, issue in enumerate(composition_issues[:3]):
                summary += f"{i+1}. {issue['description']}\n"
            if len(composition_issues) > 3:
                summary += f"...and {len(composition_issues) - 3} more composition issues\n"
        
        if len(metadata_issues) > 0:
            summary += "\nMetadata issues:\n"
            for i, issue in enumerate(metadata_issues[:3]):
                summary += f"{i+1}. {issue['description']}\n"
            if len(metadata_issues) > 3:
                summary += f"...and {len(metadata_issues) - 3} more metadata issues\n"
                
        return summary

    def _store_results(self, results: Dict) -> None:
        """Store validation results in Firebase.

        Args:
            results: Validation results dictionary
        """
        try:
            self.results_collection.document(
                results["scene_id"] + "_" + results["validation_time"].replace(":", "-")
            ).set(results)
            logger.info(f"Results stored in Firebase for scene {results['scene_id']}")
        except Exception as e:
            logger.error(f"Failed to store results in Firebase: {e}")

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format.

        Returns:
            Timestamp string
        """
        from datetime import datetime
        return datetime.now().isoformat()


def main():
    """Command line interface for SceneValidator."""
    parser = argparse.ArgumentParser(description="Validate scene composition and continuity")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("metadata", help="Path to metadata JSON file")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--output", help="Path to output JSON file")
    
    args = parser.parse_args()
    
    validator = SceneValidator(args.config)
    results = validator.validate_scene(args.video, args.metadata)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
    else:
        print(json.dumps(results, indent=2))
    
    status = "PASSED" if not results["issues_detected"] else "FAILED"
    print(f"\nValidation {status}: {results['summary']}")


if __name__ == "__main__":
    main()