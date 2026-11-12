# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Arthur Gouaze
"""limbform: open-core CAD automation for prosthetic socket fitting.

Pipeline (rule-based trim-line generation for transtibial sockets):

    io.load_mesh -> io.normalize_orientation
        -> landmarks.place_landmarks
        -> trimline.generate_trimline
        -> io.export_mesh / io.export_curve
"""

__version__ = "0.0.1"

__all__ = ["__version__"]
