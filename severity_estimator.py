"""
Severity Estimator
==================
Heuristic-based severity estimation for detected road defects.
Uses bounding box dimensions relative to image size, with
class-specific thresholds.

Severity Levels:
  - Low:    Minor defect, small area/length
  - Medium: Moderate defect
  - High:   Severe defect, large area/length
"""


class SeverityEstimator:
    """Estimate defect severity from bounding box geometry."""

    # Thresholds: (low_upper, medium_upper)
    # Anything above medium_upper is "High"
    THRESHOLDS = {
        "Pothole": {
            "metric": "area_ratio",       # bbox area / image area
            "low_upper": 0.02,            # < 2% → Low
            "medium_upper": 0.05,         # 2-5% → Medium, > 5% → High
        },
        "LongitudinalCrack": {
            "metric": "height_ratio",     # bbox height / image height
            "low_upper": 0.10,            # < 10% → Low
            "medium_upper": 0.25,         # 10-25% → Medium, > 25% → High
        },
        "TransverseCrack": {
            "metric": "width_ratio",      # bbox width / image width
            "low_upper": 0.10,            # < 10% → Low
            "medium_upper": 0.25,         # 10-25% → Medium, > 25% → High
        },
        "AlligatorCrack": {
            "metric": "area_ratio",       # bbox area / image area
            "low_upper": 0.03,            # < 3% → Low
            "medium_upper": 0.08,         # 3-8% → Medium, > 8% → High
        },
    }

    # Color scheme for visualization (BGR for OpenCV)
    SEVERITY_COLORS_BGR = {
        "Low": (0, 200, 0),        # Green
        "Medium": (0, 165, 255),   # Orange
        "High": (0, 0, 255),       # Red
    }

    # Color scheme (RGB for matplotlib)
    SEVERITY_COLORS_RGB = {
        "Low": (0, 200/255, 0),
        "Medium": (1, 165/255, 0),
        "High": (1, 0, 0),
    }

    def estimate(
        self,
        bbox: tuple[float, float, float, float],
        class_name: str,
        image_width: int,
        image_height: int,
    ) -> str:
        """
        Estimate severity of a detected defect.

        Args:
            bbox: (x1, y1, x2, y2) in pixel coordinates.
            class_name: One of 'Pothole', 'LongitudinalCrack',
                        'AlligatorCrack', 'TransverseCrack'.
            image_width: Width of the source image in pixels.
            image_height: Height of the source image in pixels.

        Returns:
            Severity string: 'Low', 'Medium', or 'High'.
        """
        x1, y1, x2, y2 = bbox
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        bbox_area = bbox_w * bbox_h
        img_area = image_width * image_height

        thresholds = self.THRESHOLDS.get(class_name)
        if thresholds is None:
            # Fallback: generic area-based estimation
            ratio = bbox_area / img_area if img_area > 0 else 0
            if ratio < 0.02:
                return "Low"
            elif ratio < 0.05:
                return "Medium"
            return "High"

        metric = thresholds["metric"]
        if metric == "area_ratio":
            value = bbox_area / img_area if img_area > 0 else 0
        elif metric == "height_ratio":
            value = bbox_h / image_height if image_height > 0 else 0
        elif metric == "width_ratio":
            value = bbox_w / image_width if image_width > 0 else 0
        else:
            value = bbox_area / img_area if img_area > 0 else 0

        if value < thresholds["low_upper"]:
            return "Low"
        elif value < thresholds["medium_upper"]:
            return "Medium"
        else:
            return "High"

    def get_color_bgr(self, severity: str) -> tuple[int, int, int]:
        """Get BGR color for a severity level (for OpenCV)."""
        return self.SEVERITY_COLORS_BGR.get(severity, (255, 255, 255))

    def get_color_rgb(self, severity: str) -> tuple[float, float, float]:
        """Get RGB color for a severity level (for matplotlib)."""
        return self.SEVERITY_COLORS_RGB.get(severity, (1, 1, 1))
