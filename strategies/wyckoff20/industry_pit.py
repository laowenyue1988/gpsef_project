"""
Point-in-Time (PiT) Industry Classification Module.

P0-2 FIX: Ensures that when building snapshots for date T,
only industry classifications published on or before T are used.
This prevents look-ahead bias from future industry reclassifications.

Data source: data/performance_reports/*.parquet (same as fundamental_ml_v2)
"""

import os
import logging
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Resolve project root (strategies/wyckoff20/industry_pit.py -> project root)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_PERF_DIR = os.path.join(_PROJECT_ROOT, 'data', 'performance_reports')

# Column name candidates for industry
_INDUSTRY_COL_CANDIDATES = ['所处行业', 'industry', '行业']
# Column name substring for publish/announcement date
_PUB_DATE_SUBSTRING = '公告'


class IndustryMapPiT:
    """
    Point-in-Time industry classification.

    Loads all performance_reports parquet files, extracts (code, industry, pub_date),
    and for any query date T returns the industry map using only reports published <= T.
    """

    def __init__(self, perf_dir: str = None):
        self._perf_dir = perf_dir or _PERF_DIR
        self._all_records: Optional[pd.DataFrame] = None  # columns: code, industry, pub_date
        self._cache: Dict[str, Dict[str, str]] = {}  # date_str -> {code: industry}
        self._loaded = False

    def _load_all(self):
        """Load and concatenate all performance_reports into a single DataFrame."""
        if self._loaded:
            return

        if not os.path.isdir(self._perf_dir):
            logger.warning(f"Performance reports directory not found: {self._perf_dir}")
            self._all_records = pd.DataFrame(columns=['code', 'industry', 'pub_date'])
            self._loaded = True
            return

        perf_files = sorted([f for f in os.listdir(self._perf_dir) if f.endswith('.parquet')])
        if not perf_files:
            logger.warning("No performance report parquet files found")
            self._all_records = pd.DataFrame(columns=['code', 'industry', 'pub_date'])
            self._loaded = True
            return

        all_dfs = []
        for fname in perf_files:
            fpath = os.path.join(self._perf_dir, fname)
            try:
                df = pd.read_parquet(fpath)
                if 'code' not in df.columns:
                    continue

                # Find industry column
                ind_col = None
                for candidate in _INDUSTRY_COL_CANDIDATES:
                    if candidate in df.columns:
                        ind_col = candidate
                        break
                if ind_col is None:
                    continue

                # Find publish date column (contains '公告')
                pub_col = None
                for col in df.columns:
                    if _PUB_DATE_SUBSTRING in str(col):
                        pub_col = col
                        break
                if pub_col is None:
                    # Fallback: try to infer date from filename
                    # e.g., performance_report_20240301.parquet
                    try:
                        base = fname.replace('.parquet', '')
                        date_part = ''.join(c for c in base if c.isdigit())
                        if len(date_part) >= 8:
                            file_date = pd.Timestamp(datetime.strptime(date_part[:8], '%Y%m%d'))
                            df['_pub_date'] = file_date
                            pub_col = '_pub_date'
                        else:
                            continue
                    except Exception:
                        continue

                # Extract relevant columns
                sub = df[['code', ind_col, pub_col]].copy()
                sub.columns = ['code', 'industry', 'pub_date']

                # Normalize code to 6-digit string
                sub['code'] = sub['code'].astype(str).str.zfill(6)

                # Filter to valid stock codes
                sub = sub[sub['code'].str.match(r'^[036]\d{5}$')]

                # Parse pub_date
                sub['pub_date'] = pd.to_datetime(sub['pub_date'], errors='coerce')
                sub = sub.dropna(subset=['pub_date'])

                # Drop rows with missing industry
                sub = sub.dropna(subset=['industry'])
                sub = sub[sub['industry'].astype(str).str.strip() != '']

                if len(sub) > 0:
                    all_dfs.append(sub[['code', 'industry', 'pub_date']])

            except Exception as e:
                logger.warning(f"Failed to read {fname}: {e}")
                continue

        if all_dfs:
            self._all_records = pd.concat(all_dfs, ignore_index=True)
            # For duplicate (code, pub_date), keep last
            self._all_records = self._all_records.sort_values('pub_date').drop_duplicates(
                subset=['code', 'pub_date'], keep='last')
            self._all_records = self._all_records.sort_values('pub_date').reset_index(drop=True)
        else:
            self._all_records = pd.DataFrame(columns=['code', 'industry', 'pub_date'])

        self._loaded = True
        logger.info(f"IndustryMapPiT: loaded {len(self._all_records)} records "
                     f"from {len(all_dfs)} files, "
                     f"date range: {self._all_records['pub_date'].min()} ~ "
                     f"{self._all_records['pub_date'].max()}")

    def get_map(self, date) -> Dict[str, str]:
        """
        Get industry map for a given date (Point-in-Time).

        For each stock code, returns the industry classification from the
        most recent report published on or before `date`.

        Args:
            date: datetime, Timestamp, or date string (YYYYMMDD / YYYY-MM-DD)

        Returns:
            Dict mapping 6-digit code -> industry name
        """
        self._load_all()

        if self._all_records.empty:
            return {}

        # Normalize date
        if isinstance(date, str):
            date = pd.Timestamp(date)
        elif isinstance(date, datetime):
            date = pd.Timestamp(date)
        elif not isinstance(date, pd.Timestamp):
            date = pd.Timestamp(date)

        # Use date only (no time component) for comparison
        query_date = date.normalize()

        # Check cache
        cache_key = query_date.strftime('%Y%m%d')
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Filter: only records published on or before query_date
        valid = self._all_records[self._all_records['pub_date'] <= query_date]

        if valid.empty:
            self._cache[cache_key] = {}
            return {}

        # For each code, take the most recent record
        latest = valid.sort_values('pub_date').drop_duplicates(subset=['code'], keep='last')

        result = dict(zip(latest['code'], latest['industry']))
        self._cache[cache_key] = result

        return result

    def get_map_for_date_str(self, date_str: str) -> Dict[str, str]:
        """Convenience method accepting YYYYMMDD string."""
        return self.get_map(date_str)
