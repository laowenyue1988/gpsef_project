# -*- coding: utf-8 -*-
"""
Wyckoff 2.0 V10 - GPSEF Framework Package

40-factor Wyckoff system with Purged K-Fold CV + 5-seed ensemble +
Isotonic calibration + AUC gating + trend filter + IV position control.

Author: BULLION GPSEF Framework
Date: 2026-06-24
Version: 10.1.0
"""

from .factors_v5 import WyckoffFactorEngineV5

__all__ = ['WyckoffFactorEngineV5']
__version__ = '10.1.0'
