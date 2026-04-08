"""
utils.py
--------
Pure utility functions for the XCPD QC pipeline.
No Streamlit imports — safe to unit-test in isolation.
"""

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, NamedTuple, Tuple

import h5py
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

class QCKey(NamedTuple):
    label: str
    ses: Optional[str] = None
    task: Optional[str] = None
    run: Optional[str] = None


@dataclass
class QCEntry:
    svg_list: List[Path]
    qc_name: str
    metric_name: str
    file_type: Literal["svg", "hdf5", "tsv"] = "svg"


def load_qc_configs(pipeline_name: str, config_file: str) -> List[Tuple[str, str, str, str]]:
    """
    Load QC configurations from a JSON file for the specified pipeline.

    Returns
    -------
    list of (pattern, name, id, type) tuples
    type is one of "svg", "hdf5", "tsv" (defaults to "svg" if omitted in config)
    """
    with open(config_file, "r") as f:
        configs = json.load(f)

    if pipeline_name not in configs:
        raise ValueError(f"Pipeline '{pipeline_name}' not found in {config_file}.")
    pipeline_version = configs.get("pipeline", pipeline_name)                                   
    fd_config = configs.get("fd", {"threshold": 0.3, "min_minutes": 5})
    qc_configs = [
        (item["pattern"], item["name"], item["id"], item.get("type", "svg"))
        for item in configs[pipeline_name]
    ]
    return pipeline_version, qc_configs, fd_config


def load_fd_params(config_file: str) -> dict:
    """
    Load FD threshold parameters from the qc_config JSON.

    Returns a dict with:
      - threshold   (float, default 0.3)
      - min_minutes (float or None — None means use 50% of total scan duration)
    """
    with open(config_file, "r") as f:
        configs = json.load(f)
    params = configs.get("fd_params", {})
    return {
        "threshold":   float(params.get("threshold", 0.3)),
        "min_minutes": float(params["min_minutes"]) if params.get("min_minutes") is not None else None,
    }


def get_qc_key(filepath: Path) -> QCKey:
    """Extract BIDS entities from a filename and return a structured QCKey."""
    fname = filepath.name

    ses  = re.search(r"ses-([a-zA-Z0-9]+)", fname)
    task = re.search(r"task-([a-zA-Z0-9]+)", fname)
    run  = re.search(r"run-([0-9]+)", fname)

    entities = {
        "ses":  f"ses-{ses.group(1)}"  if ses  else None,
        "task": task.group(1)          if task else None,
        "run":  f"run-{run.group(1)}"  if run  else None,
    }

    if entities["run"]:
        label = "run-level QC"
    elif entities["task"]:
        label = "task-level QC"
    elif entities["ses"]:
        label = "session-level QC"
    else:
        label = "subject-level QC"

    return QCKey(label=label, **entities)


_TYPE_DIRS    = {"svg": "figures", "hdf5": "func", "tsv": "func"}
_TYPE_SUFFIXES = {"svg": {".svg", ".html"}, "hdf5": {".hdf5"}, "tsv": {".tsv"}}



def _get_search_paths(base_path: Path, subdir: str) -> List[Path]:
    """Return existing subdir paths under base_path, checking flat then session-level."""
    flat = base_path / subdir
    if flat.exists():
        return [flat]
    return [
        sd / subdir
        for sd in sorted(base_path.glob("ses-*"))
        if (sd / subdir).exists()
    ]


def collect_subject_qc(
    xcpd_dir: Path,
    sub_id: str,
    configs: List[Tuple[str, str, str, str]],
) -> Dict[QCKey, List[QCEntry]]:
    """
    Walk the subject's directories and group QC files by BIDS entities + metric.
    Each config entry declares its type ("svg", "hdf5", "tsv") which controls
    which subdirectory and file suffix to search.
    """
    xcpd_dir = Path(xcpd_dir)
    base_path = xcpd_dir / f"sub-{sub_id}"

    temp_bundles: Dict[QCKey, Dict[str, QCEntry]] = defaultdict(dict)

    for pattern, qc_name, metric_id, file_type in configs:
        subdir   = _TYPE_DIRS.get(file_type, "figures")
        suffixes = _TYPE_SUFFIXES.get(file_type, {".svg", ".html"})

        for path in _get_search_paths(base_path, subdir):
            for f in sorted(path.glob(f"sub-{sub_id}*{pattern}*")):
                if f.suffix not in suffixes:
                    continue

                key = get_qc_key(f)

                if metric_id not in temp_bundles[key]:
                    temp_bundles[key][metric_id] = QCEntry(
                        svg_list=[f],
                        qc_name=qc_name,
                        metric_name=metric_id,
                        file_type=file_type,
                    )
                else:
                    if f not in temp_bundles[key][metric_id].svg_list:
                        temp_bundles[key][metric_id].svg_list.append(f)

    return {key: list(metrics.values()) for key, metrics in temp_bundles.items()}


def save_qc_results_to_csv(out_file: Path, qc_records: list) -> Path:
    """
    Upsert QC records into a CSV file (keyed on subject/session/task/run/pipeline).
    """
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for rec in qc_records:
        row = {
            "subject":  f"sub-{rec.subject_id}" if not str(rec.subject_id).startswith("sub-") else rec.subject_id,
            "session":  rec.session_id,
            "task":     rec.task_id,
            "run":      rec.run_id,
            "pipeline": rec.pipeline,
            "complete_timestamp": rec.complete_timestamp,
        }

        for m in rec.metrics:
            metric_name = m.name.replace("-", "_")
            if m.qc is not None:
                row[metric_name] = m.qc
            if hasattr(m, "value") and m.value is not None:
                row[metric_name] = m.value

        row.update({
            "require_rerun": rec.require_rerun,
            "rater":         rec.rater,
            "final_qc":      rec.final_qc,
            "notes":         next((m.notes for m in rec.metrics if m.name == "QC_notes"), None),
        })
        rows.append(row)

    df = pd.DataFrame(rows)

    if out_file.exists():
        df_existing = pd.read_csv(out_file)
        df = pd.concat([df_existing, df], ignore_index=True)

        for col in ["subject", "session", "task", "run"]:
            if col in df.columns:
                df[col] = df[col].replace({pd.NA: None, np.nan: None, "": None, "None": None})

        df = df.drop_duplicates(
            subset=["subject", "session", "task", "run", "pipeline"],
            keep="last",
        )

    end_cols = ["require_rerun", "rater", "final_qc", "notes"]
    cols = [c for c in df.columns if c not in end_cols] + [c for c in end_cols if c in df.columns]
    df = df[cols]

    df = df.sort_values(
        by=["subject", "session", "task", "run"], na_position="first"
    ).reset_index(drop=True)
    df.to_csv(out_file, index=False, na_rep="")

    return out_file


def get_metrics_from_csv(qc_results: Path):
    """
    Load an existing QC CSV and return (data_dict, get_val).

    get_val(sub_id, ses_id, task_id, run_id, metric) -> value or None
    """
    if not qc_results.exists():
        return {}, lambda *args, **kwargs: None

    df = pd.read_csv(qc_results)

    for col in ["session", "task", "run"]:
        if col not in df.columns:
            df[col] = None

    def clean_id(val, prefix):
        if pd.isna(val) or val == "":
            return None
        val_str = str(val)
        return val_str if val_str.startswith(prefix) else f"{prefix}{val_str}"

    data_dict = {}
    for _, row in df.iterrows():
        sub  = clean_id(row.get("subject"), "sub-")
        ses  = clean_id(row.get("session"), "ses-")
        task = row.get("task")
        run  = clean_id(row.get("run"), "run-")
        if pd.isna(task):
            task = None

        key = (sub, ses, task, run)
        metrics = row.drop(
            ["subject", "session", "task", "run", "pipeline", "complete_timestamp", "rater"],
            errors="ignore",
        ).to_dict()
        data_dict[key] = metrics

    def get_val(sub_id, ses_id=None, task_id=None, run_id=None, metric=None):
        if sub_id is None or metric is None:
            return None
        s_sub = sub_id if sub_id.startswith("sub-") else f"sub-{sub_id}"
        val = data_dict.get((s_sub, ses_id, task_id, run_id), {}).get(metric)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            return val
        return None

    return data_dict, get_val


# def find_coverage_files(
#     xcpd_dir: Path,
#     sub_id: str,
#     ses: Optional[str],
#     task: Optional[str],
#     run: Optional[str],
# ) -> List[Path]:
#     """
#     Find all *_stat-coverage_bold.tsv files for a given subject/session/task/run.
#     """
#     xcpd_dir = Path(xcpd_dir)
#     base = xcpd_dir / f"sub-{sub_id}"

#     func_dirs = []
#     if ses:
#         ses_label = ses if ses.startswith("ses-") else f"ses-{ses}"
#         func_dirs.append(base / ses_label / "func")
#     func_dirs.append(base / "func")

#     results = []
#     for func_dir in func_dirs:
#         if not func_dir.exists():
#             continue
#         for tsv in sorted(func_dir.glob(f"sub-{sub_id}*_stat-coverage_bold.tsv")):
#             fname = tsv.name
#             if task and f"task-{task}" not in fname:
#                 continue
#             if run:
#                 run_label = run if run.startswith("run-") else f"run-{run}"
#                 if run_label not in fname:
#                     continue
#             results.append(tsv)

#     return results


def load_coverage(sub_coverage: Path) -> Optional[Tuple[Optional[str], pd.Series]]:
    """Read a *_stat-coverage_bold.tsv and return (atlas, parcel_series).

    atlas is parsed from the filename (seg- entity); parcel_series maps
    parcel name -> numeric coverage value.  All other metadata (subject,
    session, task, run) is already known by the caller from the QCKey context.
    """
    sub_coverage = Path(sub_coverage)
    atlas = next(
        (p.split("-", 1)[1] for p in sub_coverage.stem.split("_") if p.startswith("seg-")),
        None,
    )
    try:
        df_cov = pd.read_csv(sub_coverage, sep="\t")
        # Old XCP-D (e.g. 0.7.3): long format with "Node" + "coverage" columns.
        # New XCP-D: wide format with parcel names as columns.
        if "Node" in df_cov.columns and "coverage" in df_cov.columns:
            parcels = df_cov.set_index("Node")["coverage"]
        else:
            parcels = df_cov.select_dtypes(include="number").iloc[0]
        parcels = pd.to_numeric(parcels, errors="coerce").fillna(0.0)
    except Exception as e:
        print(f"Error reading {sub_coverage}: {e}")
        return None

    return atlas, parcels


def compute_coverage_qc(
    parcels: pd.Series,
    subject: str,
    session: Optional[str],
    task: Optional[str],
    run: Optional[str],
    atlas: Optional[str],
    threshold: float = 0.5,
    fail_pct_cutoff: float = 10,
) -> dict:
    """Compute coverage QC metrics from a parcel Series returned by load_coverage."""
    values        = parcels.values.astype(float)
    n_parcels     = len(values)
    failed_mask   = values < threshold
    n_failed      = int(np.sum(failed_mask))
    pct_failed    = round((n_failed / n_parcels) * 100, 2)
    failed_labels = parcels.index[failed_mask].tolist()

    return {
        "subject":          subject,
        "session":          session,
        "task":             task,
        "run":              run,
        "atlas":            atlas,
        "n_parcels":        n_parcels,
        "n_failed_parcels": n_failed,
        "pct_failed":       pct_failed,
        "coverage_QC":      "PASS" if pct_failed < fail_pct_cutoff else "FAIL",
        "failed_parcels":   failed_labels,
    }

def save_coverage_results_to_csv(out_dir: Path, coverage_rows: list) -> list:
    """
    Upsert coverage QC rows into per-atlas CSVs inside out_dir.
    Writes two files per atlas:
      atlas-<name>_coverage_QC.csv      — subject-level summary
      atlas-<name>_failed_parcels.csv   — long format, one row per failed parcel
    Returns the list of written file paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(coverage_rows)
    if df.empty:
        return []

    id_cols = ["subject", "session", "task", "run"]
    written = []

    for atlas, atlas_df in df.groupby("atlas", sort=False):
        if atlas_df.empty:
            continue

        # --- subject-level summary (no failed_parcels column) ---
        summary_df = atlas_df.drop(columns=["failed_parcels"])
        summary_file = out_dir / f"seg-{atlas}_coverage_QC.csv"
        if summary_file.exists():
            existing = pd.read_csv(summary_file)
            summary_df = pd.concat([existing, summary_df], ignore_index=True)
            summary_df = summary_df.drop_duplicates(subset=id_cols, keep="last")
        summary_df = summary_df.sort_values(by=id_cols, na_position="first").reset_index(drop=True)
        summary_df.to_csv(summary_file, index=False, na_rep="")
        written.append(summary_file)

        # --- long-format failed parcels ---
        parcel_rows = []
        for _, row in atlas_df[atlas_df["coverage_QC"] == "FAIL"].iterrows():
            for label in row["failed_parcels"]:
                parcel_rows.append({
                    "subject":      row["subject"],
                    "session":      row["session"],
                    "task":         row["task"],
                    "run":          row["run"],
                    "parcel_label": label,
                })
        parcel_file = out_dir / f"atlas-{atlas}_failed_parcels.csv"
        if parcel_rows:
            parcel_df = pd.DataFrame(parcel_rows)
            if parcel_file.exists():
                existing = pd.read_csv(parcel_file)
                parcel_df = pd.concat([existing, parcel_df], ignore_index=True)
                parcel_df = parcel_df.drop_duplicates(subset=id_cols + ["parcel_label"], keep="last")
            parcel_df = parcel_df.sort_values(by=id_cols + ["parcel_label"], na_position="first").reset_index(drop=True)
            parcel_df.to_csv(parcel_file, index=False, na_rep="")
            written.append(parcel_file)

    return written

# def find_h5_file(
#     xcpd_dir: Path,
#     sub_id: str,
#     ses: Optional[str],
#     task: Optional[str],
#     run: Optional[str],
# ) -> Optional[Path]:
#     """
#     Locate the XCP-D HDF5 QC file for a given subject/session/task/run.
#     Searches sub-{sub_id}/[ses-{ses}/]func/ for a *_qc*.h5 file matching
#     the task and run labels.
#     """
#     xcpd_dir = Path(xcpd_dir)
#     base = xcpd_dir / f"sub-{sub_id}"

#     # Build candidate func dirs (session-level first, then flat)
#     func_dirs = []
#     if ses:
#         ses_label = ses if ses.startswith("ses-") else f"ses-{ses}"
#         func_dirs.append(base / ses_label / "func")
#     func_dirs.append(base / "func")

#     for func_dir in func_dirs:
#         if not func_dir.exists():
#             continue
#         for hdf5 in sorted(func_dir.glob(f"sub-{sub_id}*_qc*.hdf5")):
#             fname = hdf5.name
#             if task and f"task-{task}" not in fname:
#                 continue
#             if run:
#                 run_label = run if run.startswith("run-") else f"run-{run}"
#                 if run_label not in fname:
#                     continue
#             return hdf5

#     return None


def load_fd_curves(sub_h5: Path) -> dict:
    """Read FD threshold curves from an XCP-D HDF5 QC file."""
    sub_h5 = Path(sub_h5)
    parts = sub_h5.stem.split("_")
    # Getting the entities from bids file name (i.e session="01",task ="nback",run="01")
    session = next((p.split("-", 1)[1] for p in parts if p.startswith("ses-")), None)
    task    = next((p.split("-", 1)[1] for p in parts if p.startswith("task-")), None)
    run     = next((p.split("-", 1)[1] for p in parts if p.startswith("run-")), None)

    fds, ratios, remaining_frames, remaining_minutes, total_frames = [], [], [], [], []

    with h5py.File(sub_h5, "r") as f:
        base_grp = f["/dcan_motion"]
        for name in sorted(base_grp.keys(), key=lambda x: float(x.split("_")[1])):
            grp = base_grp[name]
            total     = grp["total_frame_count"][()]
            remaining = grp["remaining_total_frame_count"][()]
            fds.append(float(name.split("_")[1]))
            total_frames.append(total)
            ratios.append(remaining / total)
            remaining_frames.append(remaining)
            remaining_minutes.append(grp["remaining_seconds"][()] / 60)

    return {
        "session":           session,
        "task":              task,
        "run":               run,
        "fds":               fds,
        "ratios":            ratios,
        "remaining_frames":  remaining_frames,
        "remaining_minutes": remaining_minutes,
        "total_frames":      total_frames,
    }


def extract_fd_metrics_at_threshold(h5_path: Path, threshold: float = 1, min_duration_pct: float = 0.5) -> dict:
    """Extract remaining_minutes and total_frames at a specific FD threshold.

    PASS if remaining_minutes >= min_duration_pct * total_scan_minutes.
    """
    data = load_fd_curves(h5_path)
    idx = next((i for i, fd in enumerate(data["fds"]) if fd == threshold), None)
    if idx is None:
        return {}
    remaining = data["remaining_minutes"][idx]
    total_minutes = data["remaining_minutes"][-1]
    min_minutes = total_minutes * min_duration_pct
    return {
        "fd_threshold":      threshold,
        "remaining_minutes": round(remaining, 2),
        "total_minutes":     round(total_minutes, 2),
        "min_duration_qc":   "PASS" if remaining >= min_minutes else "FAIL",
    }


def get_current_batch(metrics_df: pd.DataFrame, current_page: int, batch_size: int):
    """Return (total_rows, current_batch_df)."""
    total_rows = len(metrics_df)
    start_idx  = (current_page - 1) * batch_size
    end_idx    = min(start_idx + batch_size, total_rows)
    return total_rows, metrics_df.iloc[start_idx:end_idx]