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
from typing import Dict, List, Optional, NamedTuple, Tuple

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


def load_qc_configs(pipeline_name: str, config_file: str) -> List[Tuple[str, str, str]]:
    """
    Load QC configurations from a JSON file for the specified pipeline.

    Returns
    -------
    list of (pattern, name, id) tuples
    """
    with open(config_file, "r") as f:
        configs = json.load(f)

    if pipeline_name not in configs:
        raise ValueError(f"Pipeline '{pipeline_name}' not found in {config_file}.")

    return [(item["pattern"], item["name"], item["id"]) for item in configs[pipeline_name]]


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


def collect_subject_qc(
    xcpd_dir: Path,
    sub_id: str,
    configs: List[Tuple[str, str, str]],
) -> Dict[QCKey, List[QCEntry]]:
    """
    Walk the subject's figures directory/directories and group QC files
    by BIDS entities + metric.
    """
    xcpd_dir = Path(xcpd_dir)
    base_path = xcpd_dir / f"sub-{sub_id}"
    figures_path = base_path / "figures"

    if figures_path.exists():
        search_paths = [figures_path]
    else:
        session_dirs = sorted(base_path.glob("ses-*")) if base_path.exists() else []
        search_paths = [
            sd / "figures" for sd in session_dirs if (sd / "figures").exists()
        ]

    temp_bundles: Dict[QCKey, Dict[str, QCEntry]] = defaultdict(dict)

    for pattern, qc_name, metric_id in configs:
        for path in search_paths:
            if not path.exists():
                continue

            for f in sorted(path.glob(f"sub-{sub_id}*{pattern}*")):
                if f.suffix not in {".svg", ".html"}:
                    continue

                key = get_qc_key(f)

                if metric_id not in temp_bundles[key]:
                    temp_bundles[key][metric_id] = QCEntry(
                        svg_list=[f],
                        qc_name=qc_name,
                        metric_name=metric_id,
                    )
                else:
                    if f not in temp_bundles[key][metric_id].svg_list:
                        temp_bundles[key][metric_id].svg_list.append(f)

    return {key: list(metrics.values()) for key, metrics in temp_bundles.items()}


def extract_fieldmap_method(html_path: Path) -> Dict[str, str]:
    """Parse an HTML report and return the SDC method per session."""
    results = {}

    with open(html_path, "r") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    diffusion_div = soup.find("div", id="Diffusion")
    session_divs  = diffusion_div.find_all("div", id=lambda x: x and x.startswith("ses-"))

    targets = (
        [(sd["id"], sd) for sd in session_divs]
        if session_divs
        else [("nosession", diffusion_div)]
    )

    for session_id, container in targets:
        h3 = container.find(
            "h3",
            class_="elem-title",
            string=lambda s: s and s.startswith("Susceptibility distortion correction"),
        )
        if h3:
            match = re.search(r"\((.*?)\)", h3.text)
            method = match.group(1) if match else "UNKNOWN"
        else:
            method = "NOT FOUND"

        results[session_id] = method

    return results


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


def get_current_batch(metrics_df: pd.DataFrame, current_page: int, batch_size: int):
    """Return (total_rows, current_batch_df)."""
    total_rows = len(metrics_df)
    start_idx  = (current_page - 1) * batch_size
    end_idx    = min(start_idx + batch_size, total_rows)
    return total_rows, metrics_df.iloc[start_idx:end_idx]