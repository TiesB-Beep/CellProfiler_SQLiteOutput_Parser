#!/usr/bin/env python3
"""
Production-grade SQLite analysis script for _Per_image tables.
Extracts, cleans, joins genotype data, and generates curated features with boxplots.
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import datetime as _dt

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, inspect, text


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

GENOTYPE_DISPLAY_ORDER = ['ioN', 'ioA', 'phA']
AREA_CONVERSION_FACTOR = 0.11669056  # Pixel area to square micrometer conversion



# --- WELL DISCOVERY / NORMALIZATION ---
WELL_COL_CANDIDATES = [
    "Image_Metadata_Well", "Metadata_Well", "Well", "well", 
    "Image_WellID", "Image_Well", "WellID"
]
ROW_COL_CANDIDATES = [
    ("Row", "Col"), ("WellRow", "WellCol"), 
    ("Image_Metadata_Row", "Image_Metadata_Col")
]


def normalize_well(val: str, plate_size_hint: Optional[int] = None) -> Optional[str]:
    """Return canonical Well like 'A01'. Accept 'a1','A1','A01','A001'. plate_size_hint: 96 or 384."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip().upper().replace(" ", "")
    # Patterns: 'A1','A01','A001'
    m = re.match(r"^([A-Z])0*([0-9]+)$", s)
    if not m:  # already like A01?
        if re.match(r"^[A-Z][0-9]{2}$", s): 
            return s
        return None
    row, col = m.group(1), int(m.group(2))
    if plate_size_hint == 384:
        if row < "A" or row > "P": 
            return None
        if not (1 <= col <= 24): 
            return None
        return f"{row}{col:02d}"
    # default assume 96
    if row < "A" or row > "H": 
        # allow 384 rows but still normalize; no hard fail
        return f"{row}{col:02d}"
    return f"{row}{col:02d}"


def discover_or_build_well(df: pd.DataFrame) -> Optional[pd.Series]:
    """Return a canonical 'Well' Series if possible; else None."""
    for c in WELL_COL_CANDIDATES:
        if c in df.columns:
            return df[c].apply(lambda v: normalize_well(v))
    # Try build from Row/Col
    for r, c in ROW_COL_CANDIDATES:
        if r in df.columns and c in df.columns:
            def _mk(row):
                rch = str(row[r]).strip().upper()[0] if pd.notna(row[r]) else None
                try: 
                    cc = int(row[c])
                except: 
                    cc = None
                return normalize_well(f"{rch}{cc}" if rch and cc else None)
            return df.apply(_mk, axis=1)
    return None


# --- GENOTYPE DISCOVERY ---
def detect_genotype_col(df: pd.DataFrame) -> Optional[str]:
    exact = "Image_Metadata_Genotype"
    if exact in df.columns: 
        return exact
    for c in df.columns:
        if "genotype" in c.lower():
            return c
    return None


# --- JOIN KEYS DISCOVERY ---
PREFERRED_KEYS = [
    ["ImageNumber"],
    ["Image_Metadata_Plate", "Image_Metadata_Well", "Image_Metadata_Site"],
    ["Image_Metadata_Plate", "Image_Metadata_Well"],
    ["PlateID", "Well", "Site"],
    ["PlateID", "Well"],
    ["Well"],
]


def add_standard_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Add standard 'Well' if derivable; mirror PlateID into Image_Metadata_Plate if needed."""
    out = df.copy()
    if "Well" not in out.columns:
        well_series = discover_or_build_well(out)
        if well_series is not None:
            out["Well"] = well_series
    # unify plate id alias
    plate_aliases = [c for c in out.columns if c.lower() in {
        "image_metadata_plate", "image_metadata_plateid", "plateid", "metadata_plateid"
    }]
    if plate_aliases and "Image_Metadata_Plate" not in out.columns:
        out["Image_Metadata_Plate"] = out[plate_aliases[0]]
    return out


def is_unique(df: pd.DataFrame, keys) -> bool:
    return df.duplicated(subset=keys).sum() == 0


def safe_left_join(left: pd.DataFrame, right: pd.DataFrame, right_cols,
                   candidate_keys=None, min_coverage=0.95, strict=False, logger=None):
    left = add_standard_keys(left)
    right = add_standard_keys(right)
    candidate_keys = candidate_keys or PREFERRED_KEYS
    right_cols = list(dict.fromkeys(right_cols))  # de-dupe, keep order
    tried = []
    
    for keys in candidate_keys:
        if not (all(k in left.columns for k in keys) and all(k in right.columns for k in keys)):
            continue
        tried.append(keys)
        r_slim = right[keys + right_cols].copy()
        
        if not is_unique(r_slim, keys):
            # try collapse only if values per key are identical
            aggdict = {c: (lambda s: s.iloc[0] if s.nunique(dropna=False) == 1 else pd.NA) 
                      for c in right_cols}
            r_slim = r_slim.groupby(keys, dropna=False).agg(aggdict).reset_index()
            if any(r_slim[c].isna().any() for c in right_cols):
                if strict: 
                    raise ValueError(f"Join conflicts on keys {keys}; multiple non-identical values.")
        
        merged = left.merge(r_slim, on=keys, how="left", validate="many_to_one")
        cov = 1.0 - merged[right_cols[0]].isna().mean()
        if logger: 
            logger.info(f"Join try keys={keys} coverage={cov:.1%}")
        
        if cov >= min_coverage or not strict:
            if cov < min_coverage and logger: 
                logger.warning(f"Coverage {cov:.1%} < {min_coverage:.0%}; proceeding with NaNs.")
            return merged, keys, cov, tried
    
    # no viable
    if strict:
        raise ValueError(f"No viable join; tried {tried}.")
    return left, None, 0.0, tried


# --- PLATEID PARSING ---
PLATE_CANDIDATES = ["Image_Metadata_Plate", "Image_Metadata_PlateID", "PlateID", "Metadata_PlateID"]


def _ddmmyy_to_iso(d):
    # '250722' -> '2022-07-25' (assume 2000s for yy<70 else 1900s)
    day = int(d[0:2])
    mon = int(d[2:4]) 
    yy = int(d[4:6])
    year = 2000 + yy if yy < 70 else 1900 + yy
    return _dt.date(year, mon, day).isoformat()


PLATE_PATTERNS = [
    re.compile(r"^(?P<d6>\d{6})-(?P<proj>[^-]+)-(?P<batch>[^-]+)$"),                # 250722-Mix1-MetOH
    re.compile(r"^(?P<d8>\d{8})[-_](?P<proj>[^-_]+)[-_](?P<batch>[^-_]+)$"),        # 20220725-Proj-Batch
    re.compile(r"^(?P<proj>[^-_]+)[-_](?P<batch>[^-_]+)[-_](?P<d8>\d{8}).*$"),      # Proj-Batch-YYYYMMDD...
]


def infer_metadata(df: pd.DataFrame, logger=None) -> pd.DataFrame:
    out = df.copy()
    if all(k in out.columns for k in ["Image_Metadata_Date", "Image_Metadata_Project", "Image_Metadata_Batch"]):
        return out
    
    plate_col = next((c for c in PLATE_CANDIDATES if c in out.columns), None)
    if "Image_Metadata_Date" not in out.columns:
        out["Image_Metadata_Date"] = pd.NA
    if "Image_Metadata_Project" not in out.columns:
        out["Image_Metadata_Project"] = pd.NA
    if "Image_Metadata_Batch" not in out.columns:
        out["Image_Metadata_Batch"] = pd.NA
    
    if not plate_col:
        if logger: 
            logger.warning("No PlateID-like column found to parse metadata.")
        return out
    
    def _parse(s):
        if pd.isna(s): 
            return (pd.NA, pd.NA, pd.NA)
        s = str(s)
        for pat in PLATE_PATTERNS:
            m = pat.match(s)
            if m:
                if "d6" in m.groupdict():
                    return (_ddmmyy_to_iso(m.group("d6")), m.group("proj"), m.group("batch"))
                if "d8" in m.groupdict():
                    d8 = m.group("d8")
                    return (f"{d8[0:4]}-{d8[4:6]}-{d8[6:8]}", m.group("proj"), m.group("batch"))
        return (pd.NA, pd.NA, pd.NA)
    
    dates, projs, batches = zip(*out[plate_col].apply(_parse))
    out["Image_Metadata_Date"] = list(dates)
    out["Image_Metadata_Project"] = list(projs) 
    out["Image_Metadata_Batch"] = list(batches)
    
    # warn if low parse rate
    pr = 1.0 - pd.Series(dates).isna().mean()
    if logger and pr < 0.9: 
        logger.warning(f"Parsed metadata from PlateID for {pr:.1%} rows; check PlateID patterns.")
    return out


def winsorize(series: pd.Series, lower_pct=1, upper_pct=99) -> pd.Series:
    """Cap values at specified percentiles for plotting."""
    q_lower = series.quantile(lower_pct / 100.0)
    q_upper = series.quantile(upper_pct / 100.0)
    return series.clip(lower=q_lower, upper=q_upper)


def convert_area_columns(
    df: pd.DataFrame,
    factor: float = AREA_CONVERSION_FACTOR,
) -> pd.DataFrame:
    """Multiply *_AreaShape_Area columns by the pixel-to-um^2 factor."""
    area_cols = [col for col in df.columns if col.endswith('_AreaShape_Area')]
    if not area_cols:
        return df
    converted = df.copy()
    converted_cols = []
    for col in area_cols:
        numeric_values = pd.to_numeric(converted[col], errors='coerce')
        if numeric_values.notna().any():
            converted[col] = numeric_values * factor
            converted_cols.append(col)
        else:
            converted[col] = numeric_values
    if converted_cols:
        logger.info(
            f"Converted {len(converted_cols)} AreaShape area columns to square micrometers."
        )
    return converted

def order_genotypes(genotype_values: List[Any]) -> List[Any]:
    """Return genotypes in predefined order when available, others alphabetically."""
    if not genotype_values:
        return []

    desired = [g.lower() for g in GENOTYPE_DISPLAY_ORDER]

    def sort_key(value):
        label = str(value)
        lower = label.lower()
        try:
            return (0, desired.index(lower))
        except ValueError:
            return (1, label)

    return sorted(genotype_values, key=sort_key)


def prepare_plot_dataframe(
    df: pd.DataFrame,
    feature: str,
    genotype_col: str,
) -> pd.DataFrame:
    """Return dataframe filtered to rows with valid feature and genotype values."""
    return df.dropna(subset=[feature, genotype_col]).copy()


def build_replicate_summary(
    plot_df: pd.DataFrame,
    genotype_col: str,
    feature: str,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], Optional[str]]:
    '''Return per-replicate averages for strip overlays, counts, and unit name.'''
    if plot_df is None or plot_df.empty:
        return None, None, None

    missing_essentials = [col for col in (feature, genotype_col) if col not in plot_df.columns]
    if missing_essentials:
        return None, None, None

    replicate_cols: List[str] = []
    replicate_unit: Optional[str] = None

    well_col = 'Well' if 'Well' in plot_df.columns else 'Image_Metadata_Well' if 'Image_Metadata_Well' in plot_df.columns else None
    if well_col:
        replicate_cols.append(well_col)
        replicate_unit = 'well'

    if not replicate_cols:
        for candidate in ['Image_Metadata_Replicate', 'Metadata_Replicate', 'Replicate']:
            if candidate in plot_df.columns:
                replicate_cols.append(candidate)
                replicate_unit = 'replicate'
                break

    if not replicate_cols:
        return None, None, None

    plate_col = next((c for c in ['Image_Metadata_Plate', 'Image_Metadata_PlateID', 'PlateID', 'Metadata_PlateID'] if c in plot_df.columns), None)

    grouping_cols = [genotype_col]
    if plate_col:
        grouping_cols.append(plate_col)
    grouping_cols.extend(replicate_cols)

    valid = plot_df.dropna(subset=[feature, genotype_col] + replicate_cols)
    if valid.empty:
        return None, None, None

    summarized = (
        valid.groupby(grouping_cols, dropna=False)[feature]
        .mean()
        .reset_index()
    )

    replicate_counts = summarized.groupby(genotype_col).size()

    return summarized, replicate_counts, replicate_unit



def compute_feature_y_limits(
    plot_jobs: List[Dict[str, Any]],
    use_winsorize: bool = False,
    lower_pct: float = 1.0,
    upper_pct: float = 99.0,
    padding_fraction: float = 0.1,
) -> Dict[str, Tuple[float, float]]:
    """Calculate per-feature y-axis limits that fully capture plotted whiskers."""
    feature_values: Dict[str, List[pd.Series]] = {}

    for job in plot_jobs:
        plot_df = job.get('plot_df')
        feature = job.get('feature')
        if plot_df is None or feature is None:
            continue
        if plot_df.empty or feature not in plot_df.columns:
            continue

        values = plot_df[feature].dropna()
        if values.empty:
            continue

        if use_winsorize:
            values = winsorize(values)

        values = pd.to_numeric(values, errors='coerce').dropna()
        if values.empty:
            continue

        feature_values.setdefault(feature, []).append(values)

    limits: Dict[str, Tuple[float, float]] = {}

    for feature, series_list in feature_values.items():
        combined = pd.concat(series_list, ignore_index=True)
        combined = pd.to_numeric(combined, errors='coerce').dropna()
        if combined.empty:
            continue

        lower_q = combined.quantile(lower_pct / 100.0)
        upper_q = combined.quantile(upper_pct / 100.0)
        if pd.isna(lower_q) or pd.isna(upper_q):
            continue

        data_min = combined.min()
        data_max = combined.max()

        if upper_q == lower_q:
            padding = abs(upper_q) * padding_fraction if upper_q != 0 else padding_fraction
        else:
            padding = (upper_q - lower_q) * padding_fraction

        lower_bound = min(lower_q - padding, data_min)
        upper_bound = max(upper_q + padding,data_max)

        if lower_bound == upper_bound:
            delta = abs(lower_bound) * padding_fraction if lower_bound != 0 else padding_fraction
            lower_bound -= delta
            upper_bound += delta

        limits[feature] = (float(lower_bound), float(upper_bound))

    return limits
def inventory_tables(engine) -> pd.DataFrame:
    """Inventory all tables with row counts and _Per_image detection."""
    inspector = inspect(engine)
    table_data = []
    
    for table_name in inspector.get_table_names():
        # Fast row count estimate
        try:
            with engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) FROM `{table_name}`"))
                row_count = result.scalar()
        except Exception as e:
            logger.error(f"Failed to count rows in {table_name}: {e}")
            row_count = 0
        
        is_per_image = bool(re.search(r'(?i)_per_image', table_name))
        table_data.append({
            'table_name': table_name,
            'row_count': row_count,
            'is_per_image': is_per_image
        })
    
    return pd.DataFrame(table_data)


def clean_dataframe(df: pd.DataFrame, regex_exclude: str) -> pd.DataFrame:
    """Remove columns matching the exclusion regex (case-insensitive)."""
    pattern = re.compile(regex_exclude, re.IGNORECASE)
    cols_to_drop = [col for col in df.columns if pattern.search(col)]
    if cols_to_drop:
        logger.info(f"Dropping {len(cols_to_drop)} columns matching regex")
    return df.drop(columns=cols_to_drop)


def load_features_from_file(filepath: str) -> List[str]:
    """Load features from newline-delimited file."""
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def create_curated_dataframe(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Create curated DataFrame with metadata columns, keys, and specified features."""
    df = convert_area_columns(df)
    # Essential columns to always include
    essential_cols = []
    metadata_cols = [col for col in df.columns if col.startswith('Image_Metadata_')]
    essential_cols.extend(metadata_cols)
    
    # Add key columns
    key_candidates = ['Well', 'Site', 'ImageNumber']
    for key_col in key_candidates:
        if key_col in df.columns:
            essential_cols.append(key_col)
    
    # Add requested features that exist
    available_features = [f for f in features if f in df.columns]
    missing_features = [f for f in features if f not in df.columns]
    
    if missing_features and available_features:
        # Only warn if some features are missing but others exist
        logger.warning(f"Some features not found in data: {missing_features}")
        logger.info(f"Available features to be included: {available_features}")
    elif missing_features and not available_features:
        # All features missing - will auto-detect later
        logger.info(f"Specified features not found, will auto-detect numeric columns during plotting")
    
    # If no specified features are available, include all numeric columns for potential plotting
    if not available_features:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        # Exclude metadata columns from auto-detection
        numeric_cols = [col for col in numeric_cols if not col.startswith('Image_Metadata_')]
        available_features = numeric_cols[:10]  # Limit to first 10 to avoid huge datasets
        if available_features:
            logger.info(f"Auto-detected numeric features: {available_features}")
    
    curated_cols = list(dict.fromkeys(essential_cols + available_features))  # Remove duplicates, preserve order
    return df[curated_cols].copy()

def create_boxplot(
    df: pd.DataFrame,
    feature: str,
    genotype_col: str,
    table_name: str,
    output_dir: Path,
    use_winsorize: bool = False,
    feature_label: str = None,
    plot_df: Optional[pd.DataFrame] = None,
    timepoint_override: Optional[str] = None,
    y_limits: Optional[Tuple[float, float]] = None,
) -> Optional[str]:
    """Create and save seaborn boxplot for a feature grouped by genotype."""

    # Extract timepoint from table name for plot title
    if timepoint_override:
        timepoint = timepoint_override
    else:
        timepoint_match = re.search(r'(T\d+)', table_name, re.IGNORECASE)
        timepoint = timepoint_match.group(1).upper() if timepoint_match else table_name

    # Prepare dataframe for plotting if not supplied
    if plot_df is None:       
        plot_df = prepare_plot_dataframe(df, feature, genotype_col) 
    else: 
        plot_df = plot_df.copy()    
        
        if plot_df.empty:        
            logger.warning(f"No valid data for plotting {feature} ({timepoint})")        
            return None

    # Log genotype distribution
    genotype_counts = plot_df[genotype_col].value_counts()
    logger.info(f"Genotype distribution for {feature} ({timepoint}): {dict(genotype_counts)}")

    # Check if we have Well column for clustering information
    has_well = 'Well' in plot_df.columns or 'Image_Metadata_Well' in plot_df.columns
    well_col = 'Well' if 'Well' in plot_df.columns else 'Image_Metadata_Well' if 'Image_Metadata_Well' in plot_df.columns else None

    if has_well and well_col:
        # Log well clustering information
        well_info = []
        for genotype in order_genotypes(list(plot_df[genotype_col].unique())):
            genotype_df = plot_df[plot_df[genotype_col] == genotype]
            n_wells = genotype_df[well_col].nunique()
            n_sites = len(genotype_df)
            well_info.append(f"{genotype}: {n_wells} wells, {n_sites} sites")
        if well_info:
            summary = '; '.join(well_info)
            logger.info(f"Well clustering for {feature} ({timepoint}): {summary}")

    # Apply winsorization for plotting only
    if use_winsorize:
        plot_df = plot_df.copy()
        plot_df[feature] = winsorize(plot_df[feature])

    # Remove negative values (and optionally zeros) that don't make biological sense (NEW)
    original_count = len(plot_df)
    plot_df = plot_df[plot_df[feature] > 0]  # or use >= 0 if zero is valid
    filtered_count = len(plot_df)

    if filtered_count < original_count:
        logger.info(f"Filtered out {original_count - filtered_count} non-positive values for {feature} ({timepoint})")

    if plot_df.empty:
        logger.warning(f"No valid positive data remaining for {feature} ({timepoint})")
        return None

    # Set up the plotting style
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 11})

    # Create the plot
    fig, ax = plt.subplots(figsize=(7, 7))

    # Create boxplot
    genotypes = order_genotypes(list(plot_df[genotype_col].unique()))
    replicate_summary, replicate_counts, replicate_unit = build_replicate_summary(plot_df, genotype_col, feature)
    use_replicate_overlay = replicate_summary is not None and not replicate_summary.empty
    strip_source = replicate_summary if use_replicate_overlay else plot_df
    strip_alpha = 0.7 if use_replicate_overlay else 0.4
    strip_size = 5 if use_replicate_overlay else 2


    sns.boxplot(
        data=plot_df,
        x=genotype_col,
        y=feature,
        order=genotypes,
        hue=genotype_col,
        hue_order=genotypes,
        palette="viridis",
        ax=ax,
        showfliers=True,
        flierprops=dict(marker='o', markersize=3, markerfacecolor='red', alpha=0.5),
        width=0.6
    )

    # Add strip plot overlay (averages per replicate when available)
    sns.stripplot(
        data=strip_source,
        x=genotype_col,
        y=feature,
        hue_order=genotypes,
        color='white',
        alpha=strip_alpha,
        size=strip_size,
        linewidth=1.0,
        edgecolor='black',
        ax=ax,
        jitter=True,
    )

    # Customize the plot
    ax.set_title(f'{feature} by Genotype - {timepoint}', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Genotype', fontsize=12, fontweight='bold')
    ax.set_ylabel(feature_label if feature_label else feature,
                  fontsize=12, fontweight='bold')

    # Rotate x-axis labels if needed
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # Apply provided y-limits for consistent scaling
    if y_limits is not None:
        ax.set_ylim(y_limits)


    # Improve layout
    plt.tight_layout()

    # Save plot with timepoint in filename
    plot_filename = f"boxplot_{timepoint}_{feature}.png"
    plot_path = output_dir / plot_filename
    plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    # Reset matplotlib settings
    plt.rcParams.update(plt.rcParamsDefault)

    return plot_filename

def create_overview_plot(all_data: dict, features: List[str], output_dir: Path,
                        feature_labels: dict = None, use_winsorize: bool = False,
                        feature_y_limits: Optional[Dict[str, Tuple[float, float]]] = None):
    """Create overview grid of features across timepoints with optional shared y-limits."""

    available_features = []
    timepoints = sorted(all_data.keys())

    for feature in features:
        exists = any(feature in all_data[tp] for tp in timepoints if all_data[tp])
        if exists:
            available_features.append(feature)

    if not available_features:
        logger.warning("No features available for overview plot")
        return

    if not timepoints:
        logger.warning("No timepoints available for overview plot")
        return

    n_features = len(available_features)
    n_timepoints = len(timepoints)
    logger.info(f"Creating overview plot: {n_features} features x {n_timepoints} timepoints")

    sns.set_style("ticks")
    plt.rcParams.update({'font.size': 11})
    fig_width = max(4 * n_timepoints, 12)
    fig_height = max(3 * n_features, 8)
    fig, axes = plt.subplots(n_features, n_timepoints, figsize=(fig_width, fig_height), dpi=150)

    if n_features == 1 and n_timepoints == 1:
        axes = [[axes]]
    elif n_features == 1:
        axes = [axes]
    elif n_timepoints == 1:
        axes = [[ax] for ax in axes]

    limits = dict(feature_y_limits) if feature_y_limits else {}
    if not limits:
        synthetic_jobs: List[Dict[str, Any]] = []
        for timepoint in timepoints:
            tp_features = all_data.get(timepoint, {}) or {}
            for feature in available_features:
                plot_df = tp_features.get(feature)
                if plot_df is None or plot_df.empty:
                    continue
                if feature not in plot_df.columns:
                    continue
                synthetic_jobs.append({'feature': feature, 'plot_df': plot_df})

        if synthetic_jobs:
            limits = compute_feature_y_limits(
                synthetic_jobs,
                use_winsorize=use_winsorize,
                lower_pct=1.0,
                upper_pct=99.0,
                padding_fraction=0.05,
            )

    for i, feature in enumerate(available_features):
        for j, timepoint in enumerate(timepoints):
            ax = axes[i][j]

            if timepoint not in all_data or feature not in all_data[timepoint]:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes,
                        fontsize=12, color='gray')
                ax.set_title(f'{timepoint}', fontsize=10, fontweight='bold')
                continue

            plot_df = all_data[timepoint][feature].copy()
            if plot_df.empty:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes,
                        fontsize=12, color='gray')
                ax.set_title(f'{timepoint}', fontsize=10, fontweight='bold')
                continue

            if use_winsorize:
                plot_df[feature] = winsorize(plot_df[feature])

            genotype_cols = [col for col in plot_df.columns if 'genotype' in col.lower()]
            if not genotype_cols:
                ax.text(0.5, 0.5, 'No Genotype', ha='center', va='center', transform=ax.transAxes,
                        fontsize=12, color='red')
                continue

            genotype_col = genotype_cols[0]
            genotypes = order_genotypes(list(plot_df[genotype_col].unique()))

            sns.boxplot(
                data=plot_df,
                x=genotype_col,
                y=feature,
                order=genotypes,
                hue=genotype_col,
                hue_order=genotypes,
                palette="viridis",
                ax=ax,
                showfliers=True,
                flierprops=dict(marker='o', markersize=3, markerfacecolor='red', alpha=0.5),
                width=0.6
            )

            replicate_summary, _, _ = build_replicate_summary(plot_df, genotype_col, feature)
            strip_source = replicate_summary if replicate_summary is not None and not replicate_summary.empty else plot_df
            strip_alpha = 0.6 if replicate_summary is not None and not replicate_summary.empty else 0.3
            strip_size = 5 if replicate_summary is not None and not replicate_summary.empty else 1.5

            sns.stripplot(
                data=strip_source,
                x=genotype_col,
                y=feature,
                order=genotypes,
                color='white',
                linewidth=1.0,
                edgecolor='black',
                alpha=strip_alpha,
                size=strip_size,
                ax=ax,
                jitter=True
            )

            if i == 0:
                ax.set_title(f'{timepoint}', fontsize=12, fontweight='bold')
            else:
                ax.set_title('')

            if j == 0:
                feature_label = feature_labels.get(feature, feature) if feature_labels else feature
                ax.set_ylabel(feature_label, fontsize=12, fontweight='bold')
            else:
                ax.set_ylabel('')

            if i == n_features - 1:
                ax.set_xlabel('Genotype', fontsize=12)
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
            else:
                ax.set_xlabel('')
                ax.set_xticklabels([])

            limit = limits.get(feature)
            if limit:
                ax.set_ylim(limit)

            sns.despine(ax=ax)
            ax.tick_params(axis='y', labelsize=12)

    plt.tight_layout()
    plt.subplots_adjust(top=0.94, hspace=0.3, wspace=0.2)

    overview_filename = "overview_all_features_timepoints.png"
    overview_path = output_dir / overview_filename
    plt.savefig(overview_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    plt.rcParams.update(plt.rcParamsDefault)

    logger.info(f"Saved overview plot: {overview_filename}")
    return overview_filename
def main():
    parser = argparse.ArgumentParser(description="Analyze SQLite database for _Per_image tables")
    parser.add_argument("--db", required=True, help="Path to SQLite database")
    parser.add_argument("--table", help="Specific _Per_image table to plot")
    parser.add_argument("--plot-all", action="store_true", help="Create plots for all _Per_image tables found")
    parser.add_argument("--join-table", help="Table to source genotype from")
    parser.add_argument("--join-keys", nargs="+", help="Override auto-detect join keys")
    parser.add_argument("--join-cols", nargs="+", default=["Image_Metadata_Genotype"], 
                       help="Columns to join from genotype table")
    parser.add_argument("--regex-exclude", 
                       default=r".*Execution.*|.*File.*|.*Location.*|.*Frame.*|.*Group.*|.*Height.*|.*MD5Digest.*|.*Path.*|.*Scaling.*|.*Series.*|.*Threshold.*|.*URL.*|.*Width.*|.*Mass.*|.*Parent.*|.*Edge.*",
                       help="Regex pattern for columns to exclude")
    parser.add_argument("--features", nargs="+", help="Curated feature subset")
    parser.add_argument("--features-file", help="File with newline-delimited feature list")
    parser.add_argument("--feature-labels", nargs="+", help="Custom y-axis labels for features (same order as --features)")
    parser.add_argument("--labels-file", help="File with newline-delimited y-axis labels (same order as features)")
    parser.add_argument("--limit", type=int, help="Row limit for processing")
    parser.add_argument("--winsorize", action="store_true", help="Apply winsorization for plots")
    parser.add_argument("--outdir", default="./outputs", help="Output directory")
    parser.add_argument("--strict-join", action="store_true", help="Fail if join coverage below threshold")
    parser.add_argument("--min-coverage", type=float, default=0.95, help="Minimum join coverage threshold")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed")
    parser.add_argument("--example", action="store_true", help="Show example usage")
    parser.add_argument("--force-refresh", action="store_true",
                        help="Regenerate cached CSV data from the database instead of reusing previous outputs")
    
    args = parser.parse_args()
    
    if args.example:
        print("Example usage:")
        print("  python analyze_sqlite.py --db ./my.db")
        print("  python analyze_sqlite.py --db ./my.db --table Run1_Per_image --join-table PlateMap --features-file ./features.txt --winsorize")
        return
    
    # Setup output directory
    output_dir = Path(args.outdir)
    output_dir.mkdir(exist_ok=True)
    
    # Connect to database
    try:
        db_path = Path(args.db)
        if not db_path.exists():
            logger.error(f"Database file not found: {args.db}")
            sys.exit(1)
        
        engine = create_engine(f"sqlite:///{db_path}")
        logger.info(f"Connected to database: {args.db}")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        sys.exit(1)
    
    # Inventory tables
    tables_csv_path = output_dir / "tables.csv"
    tables_df = None
    if not args.force_refresh and tables_csv_path.exists():
        try:
            tables_df = pd.read_csv(tables_csv_path)
            logger.info(f"Loaded table inventory from cache: {tables_csv_path}")
        except Exception as e:
            logger.warning(f"Failed to load cached table inventory: {e}")
            tables_df = None

    if tables_df is None:
        logger.info("Inventorying tables...")
        tables_df = inventory_tables(engine)
        tables_df.to_csv(tables_csv_path, index=False)
        logger.info(f"Saved table inventory: {tables_csv_path}")
    else:
        try:
            if db_path.exists() and tables_csv_path.stat().st_mtime < db_path.stat().st_mtime:
                logger.info("Database file is newer than cached inventory; use --force-refresh if schema changed.")
        except Exception:
            pass
    
    # Filter _Per_image tables
    per_image_tables = tables_df[tables_df['table_name'].str.contains(r"T*+_Per_Image", regex=True)]['table_name'].tolist()
    logger.info(f"Found {len(per_image_tables)} _Per_image tables: {per_image_tables}")
    
    if args.dry_run:
        logger.info("Dry run - would process these tables:")
        for table in per_image_tables:
            logger.info(f"  - {table}")
        return
    
    # Load genotype table if specified
    genotype_df = None
    genotype_cols = []
    if args.join_table:
        try:
            genotype_df = pd.read_sql_table(args.join_table, engine)
            logger.info(f"Loaded genotype table {args.join_table} with {len(genotype_df)} rows")
            
            # Detect genotype columns
            genotype_cols = []
            for col in args.join_cols:
                if col in genotype_df.columns:
                    genotype_cols.append(col)
                else:
                    # Try to find genotype column
                    detected_col = detect_genotype_col(genotype_df)
                    if detected_col and detected_col not in genotype_cols:
                        genotype_cols.append(detected_col)
                        logger.info(f"Auto-detected genotype column: {detected_col}")
            
            if not genotype_cols:
                logger.warning("No genotype columns found in join table")
        except Exception as e:
            logger.error(f"Failed to load genotype table: {e}")
            if args.strict_join:
                sys.exit(1)
    
    # Determine curated features and labels
    curated_features = ["FeatureA", "FeatureB", "FeatureC"]  # Default placeholders
    feature_labels = {}  # Dictionary mapping feature names to custom labels
    
    if args.features:
        curated_features = args.features
    elif args.features_file:
        curated_features = load_features_from_file(args.features_file)
    
    # Load custom y-axis labels if provided
    if args.feature_labels:
        if len(args.feature_labels) != len(curated_features):
            logger.warning(f"Number of feature labels ({len(args.feature_labels)}) doesn't match features ({len(curated_features)})")
        else:
            feature_labels = dict(zip(curated_features, args.feature_labels))
    elif args.labels_file:
        try:
            with open(args.labels_file, 'r') as f:
                labels = [line.strip() for line in f if line.strip()]
            if len(labels) != len(curated_features):
                logger.warning(f"Number of labels in file ({len(labels)}) doesn't match features ({len(curated_features)})")
            else:
                feature_labels = dict(zip(curated_features, labels))
        except Exception as e:
            logger.error(f"Failed to load labels file: {e}")
    
    # Process each _Per_image table
    processed_tables = []
    join_results = {}
    plot_files = []
    plot_jobs = []
    
    # Dictionary to store data for overview plot: {timepoint: {feature: dataframe}}
    overview_data = {}
    
    for table_name in per_image_tables:
        logger.info(f"Processing table: {table_name}")

        clean_csv_path = output_dir / f"clean_{table_name}.csv"
        curated_csv_path = output_dir / f"curated_{table_name}.csv"

        use_cache = False
        df_clean = None

        if not args.force_refresh and clean_csv_path.exists():
            try:
                df_clean = pd.read_csv(clean_csv_path)
                use_cache = True
                logger.info(f"Reusing cached clean CSV for {table_name}")
            except Exception as e:
                logger.warning(f"Failed to load cached clean CSV for {table_name}: {e}. Refreshing from database.")
                df_clean = None
                use_cache = False

        if not use_cache:
            try:
                query = f"SELECT * FROM `{table_name}`"
                if args.limit:
                    query += f" LIMIT {args.limit}"

                df = pd.read_sql(query, engine)
                logger.info(f"Loaded {len(df)} rows from {table_name}")

                df_clean = clean_dataframe(df, args.regex_exclude)
                logger.info(f"Cleaned to {len(df_clean.columns)} columns")

                join_coverage = 0.0
                join_keys_used = None
                if genotype_df is not None and genotype_cols:
                    try:
                        df_clean, join_keys_used, join_coverage, tried_keys = safe_left_join(
                            df_clean, genotype_df, genotype_cols,
                            candidate_keys=[args.join_keys] if args.join_keys else None,
                            min_coverage=args.min_coverage,
                            strict=args.strict_join,
                            logger=logger
                        )
                        join_results[table_name] = {
                            'keys_used': join_keys_used,
                            'coverage': join_coverage,
                            'tried_keys': tried_keys,
                            'source': 'database'
                        }
                    except Exception as e:
                        logger.error(f"Join failed for {table_name}: {e}")
                        if args.strict_join:
                            continue

                df_clean.to_csv(clean_csv_path, index=False)
                logger.info(f"Saved cleaned table: {clean_csv_path}")
            except Exception as e:
                logger.error(f"Failed to process table {table_name}: {e}")
                continue

        if df_clean is None:
            logger.warning(f"No cleaned data available for {table_name}; skipping.")
            continue

        try:
            curated_df = create_curated_dataframe(df_clean, curated_features)
            curated_df.to_csv(curated_csv_path, index=False)
            if use_cache:
                logger.info(f"Refreshed curated table from cached clean data: {curated_csv_path}")
            else:
                logger.info(f"Saved curated table: {curated_csv_path}")
        except Exception as e:
            logger.error(f"Failed to create curated dataframe for {table_name}: {e}")
            continue

        processed_tables.append(table_name)

        if use_cache and table_name not in join_results:
            join_results[table_name] = {
                'keys_used': None,
                'coverage': None,
                'tried_keys': [],
                'source': 'cache'
            }

        should_plot = (args.table and table_name == args.table) or args.plot_all
        if should_plot:
            genotype_col = None
            if genotype_cols:
                genotype_col = next((col for col in genotype_cols if col in curated_df.columns), None)
            if not genotype_col:
                genotype_col = detect_genotype_col(curated_df)

            if genotype_col and genotype_col in curated_df.columns:
                plot_features = [f for f in curated_features if f in curated_df.columns]
                if not plot_features:
                    plot_features = curated_df.select_dtypes(include=['number']).columns.tolist()
                    plot_features = [f for f in plot_features if not f.startswith('Image_Metadata_')]
                    logger.info(f"Auto-detected features for plotting {table_name}: {plot_features}")
                else:
                    logger.info(f"Using specified features for plotting {table_name}: {plot_features}")

                logger.info(f"Creating plots for {table_name} with {len(plot_features)} features")

                for feature in plot_features:
                    try:
                        custom_label = feature_labels.get(feature, None)
                        timepoint_match = re.search(r'(T\d+)', table_name, re.IGNORECASE)
                        timepoint = timepoint_match.group(1).upper() if timepoint_match else table_name

                        base_columns = [feature, genotype_col]
                        extra_columns = []
                        for col in ('Well', 'Image_Metadata_Well'):
                            if col in curated_df.columns and col not in base_columns:
                                extra_columns.append(col)
                        columns_for_plot = base_columns + extra_columns
                        df_for_plot = curated_df[columns_for_plot].copy()

                        plot_df = prepare_plot_dataframe(df_for_plot, feature, genotype_col)
                        overview_data.setdefault(timepoint, {})[feature] = plot_df.copy()

                        plot_jobs.append({
                            'table_name': table_name,
                            'feature': feature,
                            'genotype_col': genotype_col,
                            'feature_label': custom_label,
                            'plot_df': plot_df,
                            'source_df': df_for_plot,
                            'timepoint': timepoint,
                        })
                    except Exception as e:
                        logger.error(f"Failed to prepare plot data for {feature} in {table_name}: {e}")
            else:
                logger.warning(f"Cannot create plots for {table_name}: no genotype column available")
    feature_limits: Dict[str, Tuple[float, float]] = {}

    for job in plot_jobs:
        try:
            plot_filename = create_boxplot(
                job['source_df'],
                job['feature'],
                job['genotype_col'],
                job['table_name'],
                output_dir,
                args.winsorize,
                job['feature_label'],
                plot_df=job['plot_df'],
                timepoint_override=job['timepoint']
            )
            if plot_filename:
                plot_files.append(plot_filename)
        except Exception as e:
            logger.error(f"Failed to create plot for {job['feature']} in {job['table_name']}: {e}")

    # Create comprehensive overview plot if we have data and plotting was requested
    if overview_data and (args.plot_all or args.table):
        logger.info(f"Overview data available for timepoints: {list(overview_data.keys())}")
        try:
            # Get the features that were actually plotted
            all_features = set()
            for tp, tp_data in overview_data.items():
                logger.info(f"Timepoint {tp} has features: {list(tp_data.keys())}")
                all_features.update(tp_data.keys())
            all_features = sorted(list(all_features))
            
            logger.info(f"All features for overview: {all_features}")
            
            if all_features:
                logger.info("Creating comprehensive overview plot...")
                overview_file = create_overview_plot(
                    overview_data,
                    all_features,
                    output_dir,
                    feature_labels,
                    args.winsorize,
                )
                if overview_file:
                    plot_files.append(overview_file)
                    logger.info(f"Overview plot created successfully: {overview_file}")
                else:
                    logger.warning("Overview plot function returned None")
            else:
                logger.warning("No features available for overview plot")
        except Exception as e:
            logger.error(f"Failed to create overview plot: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    else:
        logger.info(f"Overview plot skipped. Data available: {bool(overview_data)}, "
                   f"Plot all: {args.plot_all}, Table specified: {bool(args.table)}")
        if overview_data:
            logger.info(f"Available timepoints: {list(overview_data.keys())}")
            for tp, features in overview_data.items():
                logger.info(f"  {tp}: {len(features)} features")
    
    # Generate run report
    report = {
        'timestamp': _dt.datetime.now().isoformat(),
        'database': str(args.db),
        'library_versions': {
            'pandas': pd.__version__,
            'matplotlib': plt.matplotlib.__version__,
            'sqlalchemy': '2.x'  # Placeholder - actual version detection would require import
        },
        'parameters': {
            'regex_exclude': args.regex_exclude,
            'curated_features': curated_features,
            'winsorize': args.winsorize,
            'min_coverage': args.min_coverage,
            'strict_join': args.strict_join,
            'area_conversion_factor': AREA_CONVERSION_FACTOR
        },
        'results': {
            'total_tables_found': len(tables_df),
            'per_image_tables_found': len(per_image_tables),
            'tables_processed': len(processed_tables),
            'processed_table_names': processed_tables,
            'join_results': join_results,
            'plot_files_created': plot_files
        }
    }
    
    report_path = output_dir / "run_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Summary
    logger.info("="*50)
    logger.info("ANALYSIS COMPLETE")
    logger.info(f"Tables found: {len(tables_df)}")
    logger.info(f"_Per_image tables: {len(per_image_tables)}")
    logger.info(f"Tables processed: {len(processed_tables)}")
    logger.info(f"Plots created: {len(plot_files)}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Run report: {report_path}")
    logger.info("="*50)


if __name__ == "__main__":
    main()




























































