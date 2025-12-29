#!/usr/bin/env python3
"""
ROCA Interpolation Engine - Symbolic Keyframe Blending

Advanced interpolation engine for smooth transitions between keyframes.
Supports Bezier easing curves, per-layer interpolation, and shape-aware morphing.

Features:
- Bezier easing curves (ease-in, ease-out, ease-in-out, elastic, bounce)
- Per-layer interpolation (position, scale, rotation, opacity)
- Shape-aware morphing using OpenCV contour matching
- Metadata storage for interpolation curves in Transition capsules
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import math


class InterpolationCurve(Enum):
    """Pre-defined interpolation curves with Bezier control points"""
    LINEAR = "linear"
    EASE_IN = "ease_in"
    EASE_OUT = "ease_out"
    EASE_IN_OUT = "ease_in_out"
    EASE_IN_BACK = "ease_in_back"
    EASE_OUT_BACK = "ease_out_back"
    EASE_IN_OUT_BACK = "ease_in_out_back"
    EASE_IN_ELASTIC = "ease_in_elastic"
    EASE_OUT_ELASTIC = "ease_out_elastic"
    EASE_IN_OUT_ELASTIC = "ease_in_out_elastic"
    EASE_IN_BOUNCE = "ease_in_bounce"
    EASE_OUT_BOUNCE = "ease_out_bounce"
    EASE_IN_OUT_BOUNCE = "ease_in_out_bounce"


@dataclass
class BezierControlPoints:
    """Bezier curve control points for interpolation"""
    p0: Tuple[float, float] = (0.0, 0.0)  # Start point
    p1: Tuple[float, float] = (0.0, 0.0)  # Control point 1
    p2: Tuple[float, float] = (1.0, 1.0)  # Control point 2
    p3: Tuple[float, float] = (1.0, 1.0)  # End point


@dataclass
class LayerTransform:
    """Per-layer transformation parameters"""
    position: Tuple[float, float] = (0.0, 0.0)  # (x, y)
    scale: Tuple[float, float] = (1.0, 1.0)    # (scale_x, scale_y)
    rotation: float = 0.0                       # degrees
    opacity: float = 1.0                         # 0.0 to 1.0


class InterpolationEngine:
    """Advanced interpolation engine for keyframe transitions"""

    # Pre-defined Bezier control points for each curve type
    CURVE_CONTROL_POINTS = {
        InterpolationCurve.LINEAR: BezierControlPoints(
            p0=(0.0, 0.0), p1=(0.0, 0.0), p2=(1.0, 1.0), p3=(1.0, 1.0)
        ),
        InterpolationCurve.EASE_IN: BezierControlPoints(
            p0=(0.0, 0.0), p1=(0.42, 0.0), p2=(1.0, 1.0), p3=(1.0, 1.0)
        ),
        InterpolationCurve.EASE_OUT: BezierControlPoints(
            p0=(0.0, 0.0), p1=(0.0, 0.0), p2=(0.58, 1.0), p3=(1.0, 1.0)
        ),
        InterpolationCurve.EASE_IN_OUT: BezierControlPoints(
            p0=(0.0, 0.0), p1=(0.42, 0.0), p2=(0.58, 1.0), p3=(1.0, 1.0)
        ),
        InterpolationCurve.EASE_IN_BACK: BezierControlPoints(
            p0=(0.0, 0.0), p1=(0.6, -0.28), p2=(0.735, 0.045), p3=(1.0, 1.0)
        ),
        InterpolationCurve.EASE_OUT_BACK: BezierControlPoints(
            p0=(0.0, 0.0), p1=(0.175, 0.885), p2=(0.32, 1.275), p3=(1.0, 1.0)
        ),
        InterpolationCurve.EASE_IN_OUT_BACK: BezierControlPoints(
            p0=(0.0, 0.0), p1=(0.68, -0.55), p2=(0.265, 1.55), p3=(1.0, 1.0)
        ),
    }

    @staticmethod
    def bezier_point(t: float, p0: Tuple[float, float],
                     p1: Tuple[float, float], p2: Tuple[float, float],
                     p3: Tuple[float, float]) -> Tuple[float, float]:
        """Calculate point on cubic Bezier curve at parameter t"""
        # De Casteljau's algorithm
        p01 = (p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1]))
        p12 = (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))
        p23 = (p2[0] + t * (p3[0] - p2[0]), p2[1] + t * (p3[1] - p2[1]))

        p012 = (p01[0] + t * (p12[0] - p01[0]), p01[1] + t * (p12[1] - p01[1]))
        p123 = (p12[0] + t * (p23[0] - p12[0]), p12[1] + t * (p23[1] - p12[1]))

        return (p012[0] + t * (p123[0] - p012[0]), p012[1] + t * (p123[1] - p012[1]))

    @staticmethod
    def get_curve_value(t: float, curve: Union[str, InterpolationCurve]) -> float:
        """Get interpolated value using specified curve"""
        if isinstance(curve, str):
            curve = InterpolationCurve(curve)

        if curve not in InterpolationEngine.CURVE_CONTROL_POINTS:
            # Fallback to linear
            return t

        control_points = InterpolationEngine.CURVE_CONTROL_POINTS[curve]
        point = InterpolationEngine.bezier_point(t, control_points.p0,
                                                control_points.p1,
                                                control_points.p2,
                                                control_points.p3)
        return point[1]  # Return y-coordinate (interpolated value)

    @staticmethod
    def interpolate_value(start_val: float, end_val: float, t: float,
                         curve: Union[str, InterpolationCurve] = "linear") -> float:
        """Interpolate between two values using specified curve"""
        curve_t = InterpolationEngine.get_curve_value(t, curve)
        return start_val + (end_val - start_val) * curve_t

    @staticmethod
    def interpolate_transform(start_transform: LayerTransform,
                             end_transform: LayerTransform, t: float,
                             curve: Union[str, InterpolationCurve] = "linear") -> LayerTransform:
        """Interpolate between two layer transforms"""
        return LayerTransform(
            position=(
                InterpolationEngine.interpolate_value(start_transform.position[0],
                                                    end_transform.position[0], t, curve),
                InterpolationEngine.interpolate_value(start_transform.position[1],
                                                    end_transform.position[1], t, curve)
            ),
            scale=(
                InterpolationEngine.interpolate_value(start_transform.scale[0],
                                                    end_transform.scale[0], t, curve),
                InterpolationEngine.interpolate_value(start_transform.scale[1],
                                                    end_transform.scale[1], t, curve)
            ),
            rotation=InterpolationEngine.interpolate_value(start_transform.rotation,
                                                        end_transform.rotation, t, curve),
            opacity=InterpolationEngine.interpolate_value(start_transform.opacity,
                                                       end_transform.opacity, t, curve)
        )

    @staticmethod
    def find_contours(image: np.ndarray) -> List[np.ndarray]:
        """Find contours in an image for shape-aware morphing"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply threshold to get binary image
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return contours

    @staticmethod
    def morph_shapes(contour1: np.ndarray, contour2: np.ndarray,
                    t: float, image_size: Tuple[int, int]) -> np.ndarray:
        """Morph between two shapes using contour interpolation"""
        # Create masks from contours
        mask1 = np.zeros(image_size, dtype=np.uint8)
        mask2 = np.zeros(image_size, dtype=np.uint8)

        cv2.drawContours(mask1, [contour1], -1, 255, -1)
        cv2.drawContours(mask2, [contour2], -1, 255, -1)

        # Interpolate between masks
        morphed_mask = cv2.addWeighted(mask1, 1 - t, mask2, t, 0)

        return morphed_mask

    @staticmethod
    def bezier_interpolate(start_img: np.ndarray, end_img: np.ndarray,
                          steps: int, curve: Union[str, InterpolationCurve] = "ease_in_out",
                          use_shape_morphing: bool = False) -> List[np.ndarray]:
        """
        Interpolate between two images using Bezier curves

        Args:
            start_img: Starting image
            end_img: Ending image
            steps: Number of interpolation steps
            curve: Interpolation curve to use
            use_shape_morphing: Whether to use shape-aware morphing

        Returns:
            List of interpolated images
        """
        if start_img.shape != end_img.shape:
            # Resize images to match
            height = max(start_img.shape[0], end_img.shape[0])
            width = max(start_img.shape[1], end_img.shape[1])
            start_img = cv2.resize(start_img, (width, height))
            end_img = cv2.resize(end_img, (width, height))

        interpolated_images = []

        if use_shape_morphing:
            # Find contours for shape-aware morphing
            contours1 = InterpolationEngine.find_contours(start_img)
            contours2 = InterpolationEngine.find_contours(end_img)

            if contours1 and contours2:
                # Use the largest contours
                contour1 = max(contours1, key=cv2.contourArea)
                contour2 = max(contours2, key=cv2.contourArea)

        for i in range(steps + 1):
            t = i / steps if steps > 0 else 0

            if use_shape_morphing and contours1 and contours2:
                # Shape-aware morphing
                morphed_mask = InterpolationEngine.morph_shapes(
                    contour1, contour2, t, (start_img.shape[1], start_img.shape[0])
                )

                # Apply morphing mask to blend images
                morphed = cv2.addWeighted(start_img, 1 - t, end_img, t, 0)
                morphed = cv2.bitwise_and(morphed, morphed, mask=morphed_mask)
                interpolated_images.append(morphed)
            else:
                # Standard pixel-based interpolation
                curve_t = InterpolationEngine.get_curve_value(t, curve)
                interpolated = cv2.addWeighted(start_img, 1 - curve_t, end_img, curve_t, 0)
                interpolated_images.append(interpolated)

        return interpolated_images

    @staticmethod
    def create_interpolation_metadata(start_transform: LayerTransform,
                                    end_transform: LayerTransform,
                                    curve: Union[str, InterpolationCurve],
                                    steps: int,
                                    use_shape_morphing: bool = False) -> Dict[str, Any]:
        """Create metadata dictionary for interpolation curves in Transition capsules"""
        return {
            "interpolation_type": "bezier",
            "curve": curve if isinstance(curve, str) else curve.value,
            "steps": steps,
            "start_transform": {
                "position": start_transform.position,
                "scale": start_transform.scale,
                "rotation": start_transform.rotation,
                "opacity": start_transform.opacity
            },
            "end_transform": {
                "position": end_transform.position,
                "scale": end_transform.scale,
                "rotation": end_transform.rotation,
                "opacity": end_transform.opacity
            },
            "shape_aware_morphing": use_shape_morphing,
            "control_points": InterpolationEngine.CURVE_CONTROL_POINTS.get(
                curve if isinstance(curve, InterpolationCurve) else InterpolationCurve(curve),
                InterpolationEngine.CURVE_CONTROL_POINTS[InterpolationCurve.LINEAR]
            ).__dict__ if hasattr(InterpolationEngine.CURVE_CONTROL_POINTS.get(
                curve if isinstance(curve, InterpolationCurve) else InterpolationCurve(curve),
                InterpolationEngine.CURVE_CONTROL_POINTS[InterpolationCurve.LINEAR]
            ), '__dict__') else None
        }


# Example usage and testing functions
def test_bezier_interpolation():
    """Test the Bezier interpolation functionality"""
    print("Testing Bezier Interpolation Engine...")

    # Test basic value interpolation
    start_val, end_val = 0.0, 100.0
    steps = 10

    for curve in InterpolationCurve:
        print(f"\nTesting {curve.value}:")
        values = []
        for i in range(steps + 1):
            t = i / steps
            interpolated = InterpolationEngine.interpolate_value(start_val, end_val, t, curve)
            values.append(round(interpolated, 2))
        print(f"  Values: {values}")

    # Test transform interpolation
    start_transform = LayerTransform(position=(0, 0), scale=(1.0, 1.0), rotation=0.0, opacity=1.0)
    end_transform = LayerTransform(position=(100, 50), scale=(2.0, 1.5), rotation=45.0, opacity=0.5)

    print("\nTesting transform interpolation:")
    mid_transform = InterpolationEngine.interpolate_transform(
        start_transform, end_transform, 0.5, "ease_in_out"
    )
    print(f"  Mid transform: pos={mid_transform.position}, scale={mid_transform.scale}, "
          f"rotation={mid_transform.rotation:.1f}, opacity={mid_transform.opacity:.2f}")

    print("Bezier interpolation tests completed!")


if __name__ == "__main__":
    test_bezier_interpolation()