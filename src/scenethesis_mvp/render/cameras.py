from __future__ import annotations


def camera_position(bounds: tuple[float, float, float], preset: str = "three_quarter") -> tuple[float, float, float]:
    width, depth, height = bounds
    if preset == "top":
        return (0.0, 0.0, max(width, depth) * 1.1)
    return (width * 0.55, -depth * 0.85, max(3.0, height * 1.3))


def camera_target() -> tuple[float, float, float]:
    return (0.0, 0.0, 0.7)
