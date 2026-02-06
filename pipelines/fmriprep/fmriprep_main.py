# %%
import math
import os
import pickle
import re
from argparse import ArgumentParser
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional
from functools import wraps
import numpy as np
import pandas as pd
import streamlit as st
from models import MetricQC, QCRecord
from bs4 import BeautifulSoup

# Debug: show keys on every rerun
# st.write("Session state before initialized:", st.session_state)

st.markdown('<div id="top_of_page"></div>', unsafe_allow_html=True)

def parse_args(args=None):
    parser = ArgumentParser("fmriprep QC")

    parser.add_argument(
        "--fmri_dir",
        help=(
            "The root directory of fMRI preprocessing derivatives. "
            "For example, /SCanD_project/data/local/derivatives/fmriprep/23.2.3."
        ),
        required=True,
    )
    parser.add_argument(
        "--participant_labels",
        help=("List of participants to QC"),
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        dest="out_dir",
        help="Directory to save session state and QC results",
        required=True,
    )
    return parser.parse_args(args)


args = parse_args()
# fs_metric = args.freesurfer_metric
participant_labels = args.participant_labels
fmri_dir = args.fmri_dir
out_dir = args.out_dir

participants_df = pd.read_csv(participant_labels, delimiter="\t")

st.title("fMRIPrep QC")
rater_name = st.text_input("Rater name:")
# Show the value dynamically
st.write("You entered:", rater_name)


def init_session_state():
    defaults = {
        "current_page": 1,
        "batch_size": 1,
        "current_batch_qc": {},
    }
    # Initialize defaults if not already set
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()


# Pagination
def get_current_batch(metrics_df, current_page, batch_size):
    total_rows = len(metrics_df)
    start_idx = (current_page - 1) * batch_size
    end_idx = min(start_idx + batch_size, total_rows)
    current_batch = metrics_df.iloc[start_idx:end_idx]
    return total_rows, current_batch


def save_qc_results_to_csv(out_file, qc_records):
    """
    Save QC results from Streamlit session state to a CSV file.

    Parameters
    ----------
    out_file : str or Path
        Path where the CSV will be saved.
    qc_records : list
        List of QCRecord objects (or dicts) stored.
    """
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Flatten metrics dynamically
    rows = []
    # qc_records = [QCRecord(**rec) for rec in qc_records_dict.values()]

    for rec in qc_records:
        row = {
            "subject": f"sub-{rec.subject_id}",
            "session": rec.session_id,
            "task": str(rec.task_id),
            "run": str(rec.run_id),
            "pipeline": rec.pipeline,
            "complete_timestamp": rec.complete_timestamp,
        }

        for m in rec.metrics:
            metric_name = m.name.lower().replace("-", "_")
            if m.value is not None:
                row[f"{metric_name}_value"] = m.value
            if m.qc is not None:
                row[f"{metric_name}"] = m.qc

        row.update(
            {
                "require_rerun": rec.require_rerun,
                "rater": rec.rater,
                "final_qc": rec.final_qc,
                "notes": next(
                    (m.notes for m in rec.metrics if m.name == "QC_notes"), None
                ),
            }
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    if out_file.exists():
        df_existing = pd.read_csv(out_file)
        df = pd.concat([df_existing, df], ignore_index=True)
        # Drop duplicates based on all columns or a subset
        df = df.drop_duplicates(
            subset=["subject", "session", "task", "run", "pipeline"], keep="last"
        )
    df = df.sort_values(by=["subject"]).reset_index(drop=True)
    df.to_csv(out_file, index=False)

    return out_file

def collect_qc_svgs(fmri_dir: str, sub_id: str, pattern: str) -> List[Path]:
    """
    Collect SVG QC figures.
    Returns:
      - For per-run QC:
          {
            "type": "perrun",
            "data": { ses -> task -> run -> [paths] }
          }
      
      - For global QC:
          {
            "type": "global",
            "data": [list of Path]
          }
    """
    base = Path(fmri_dir) / f"sub-{sub_id}" / "figures"
    # Collect both SVG and HTML
    search_pattern = list(base.rglob(f"sub-{sub_id}*{pattern}.*"))
    svg_paths = [f for f in search_pattern if f.suffix in {".svg", ".html"}]
    # search_pattern = f"sub-{sub_id}*{pattern}.*"
    # svg_paths = [f for f in search_pattern if f.suffix in {".svg", ".html"}]
    # svg_paths = sorted(Path(p) for p in glob(str(base / search_pattern)))

    if not svg_paths:
        st.warning(f" No {pattern} SVGs found for subject {sub_id}")
        return {
        "type": "perrun",
        "data": {}
    }
    # Try extracting entities to see if any file contains session/run/task
    has_session = False
    has_run = False
    has_task = False
    for p in svg_paths:
        e = get_entities(p)
        if e.get("session"): has_session = True
        if e.get("run"): has_run = True
        if e.get("task"): has_task = True

    # If none of them exist → GLOBAL metric
    if not (has_session or has_task or has_run):
        return {
            "type": "global",
            "data": svg_paths
        }
    
    # Otherwise → PER-RUN metric
    from collections import defaultdict
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for p in svg_paths:
        e = get_entities(p)

        ses = e.get("session") or "ses-NA"
        task = e.get("task") or "task-NA"
        run = e.get("run") or "run-NA"

        data[ses][task][run].append(p)

    return {
        "type": "perrun",
        "data": data
    }

def extract_fieldmap_method(html_path):
    """Parse a single HTML file and extract the susceptibility distortion correction method."""
    with open(html_path, "r") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    li = soup.find(
        "li",
        string=lambda s: s and "Susceptibility distortion correction" in s
    )

    if li:
        method = li.text.strip().replace("Susceptibility distortion correction: ", "")
    else:
        method = "NOT FOUND"
    return method

def get_entities(filepath: Path) -> Dict[str, Optional[str]]:
    fname = filepath.name

    sub = re.search(r"sub-([a-zA-Z0-9]+)", fname)
    ses = re.search(r"ses-([a-zA-Z0-9]+)", fname)
    task = re.search(r"task-([a-zA-Z0-9]+)", fname)
    run = re.search(r"run-([0-9]+)", fname)

    return {
        "subject": sub.group(1) if sub else None,
        "session": f"ses-{ses.group(1)}" if ses else None,
        "task": task.group(1) if task else None,
        "run": f"run-{run.group(1)}" if run else None,
    }

total_rows, current_batch = get_current_batch(
    participants_df, st.session_state.current_page, st.session_state.batch_size
)

# Save to CSV
# out_dir = "/projects/ttan/tigrbid-QC/outputs"
now = datetime.now()
# timestamp = now.strftime("%Y%m%d")  # e.g., 20250917
out_file = Path(out_dir) / f"fMRIPrep_QC_status.csv"

def get_metrics_from_csv(qc_results: Path):
    if not qc_results.exists():
        return {}, lambda *args,**kwargs: None
    
    # Load Data
    df = pd.read_csv(qc_results)

    # Standardize columns and Fill Missing Values
    for col in ['session', 'task', 'run']:
        if col not in df.columns:
            df[col] = None
    
    def clean_id (val, prefix):
        if pd.isna(val) or val == "":
            return None
        val_str = str(val)
        return val_str if val_str.startswith(prefix) else f"{prefix}{val_str}"
    
    data_dict = {}
    
    for _, row in df.iterrows():
        sub = clean_id(row.get('subject'), "sub-")
        ses = clean_id(row.get('session'), "ses-")
        task = row.get('task')
        run = clean_id(row.get('run'), "run-")
        if pd.isna(task): task = None

        # Create a specific key for this row
        key = (sub, ses, task, run)

        # Store all metrics in this row
        metrics = row.drop(['subject', 'session', 'task', 'run', 'pipeline', 'complete_timestamp', 'rater']).to_dict()
        data_dict[key] = metrics

    # Getter function
    def get_val(sub_id, ses_id=None, task_id=None, run_id=None, metric=None):
        if sub_id is None or metric is None:
            return None

        s_sub = sub_id if sub_id.startswith("sub-") else f"sub-{sub_id}"

        keys = [
            # 1. EXACT MATCH (e.g., sub-01, ses-01, task-rest, run-1)
            (s_sub, ses_id, task_id, run_id),
            
            # 2. TASK/SESSION LEVEL (Ignore Run)
            # Useful if a metric applies to the whole task in this session
            (s_sub, ses_id, task_id, None),

            # 3. SESSION LEVEL (Ignore Task & Run)
            # Useful for session notes or session-wide anatomical checks
            (s_sub, ses_id, None, None),

            # 4. GLOBAL SUBJECT LEVEL (Ignore Everything)
            # Useful for T1w QC, demographics, etc.
            (s_sub, None, None, None)
        ]

        for key in keys:
            # Look up the dict
            val = data_dict.get(key, {}).get(metric)
            if val is not None and not pd.isna(val):
                return val
        return None
    return data_dict, get_val

# Load the existing qc_results if exist
data_dict, get_val = get_metrics_from_csv(out_file)

def display_svg_group(
    svg_list: list[Path],
    sub_id: str,
    qc_name: str,
    metric_name: str,
    subject_metrics: list,
    ses=None,
    task=None,
    run=None
):
    """
    Display one or more SVGs with a QC radio button.
    ses, task, run are optional; include them to make Streamlit keys unique.
    """
    # with st.container():
    st.set_page_config(layout="wide")
    st.markdown(f"<h4> sub-{sub_id} - {qc_name} QC", unsafe_allow_html=True)
    options = ("PASS", "FAIL", "UNCERTAIN")

    for svg_path in svg_list:
        if not svg_path.exists():
            st.warning(f"Missing SVG: {svg_path.name}")
            continue
        with st.container():
            with open(svg_path, "r") as f:
                st.markdown(f.read(), unsafe_allow_html=True)
            st.write(f"**{svg_path.name}**")

        # Build unique Streamlit key
        key = f"{sub_id}"
        if ses: key += f"_{ses}"
        if task: key += f"_{task}"
        if run: key += f"_{run}"
        key += f"_{metric_name}"

        # Load existing value from CSV if present
        stored_val = get_val(
            sub_id=f"sub-{sub_id}",
            ses_id=ses,
            task_id=task,
            run_id=run,
            metric=metric_name
        )

        qc_choice = st.radio(
            f"{qc_name} QC:",
            options,
            key=key,
            label_visibility="collapsed",
            index=options.index(stored_val) if stored_val in options else None,
        )

        subject_metrics.append(MetricQC(
            name=metric_name,
            qc=qc_choice
        ))
# Collect all the current batch subject metrics
qc_records = []
for _, row in current_batch.iterrows():
    subj = row["participant_id"]
    parts = subj.split("_")
    sub_id = parts[0].split("-")[1]

    per_run_bundles = {}

    for pattern, qc_name, metric_name in [
        ("sdc_bold", "Susceptible Distortion Correction", "sdc_qc"),
        ("coreg_bold", "Coregistration", "coreg_qc"),
        ("_desc-summary_bold", "FieldMap Method", "fieldmap_method")
    ]:
        svg_bundle = collect_qc_svgs(fmri_dir, sub_id, pattern)
        for ses, task_dict in svg_bundle["data"].items():
            for task, run_dict in task_dict.items():
                for run, svg_list in run_dict.items():
                    key = (ses, task ,run)
                    per_run_bundles.setdefault(key, []).append(
                        (svg_list, qc_name, metric_name)
                    )
    for (ses, task, run), metrics_list in sorted(per_run_bundles.items()):
        st.markdown(f"### Session: {ses} | Task: {task} | Run: {run}")
        run_metrics = []
        for svg_list, qc_name, metric_name in metrics_list:
            if metric_name == "fieldmap_method":
                for html_path in svg_list:
                    method = extract_fieldmap_method(html_path)
                    run_metrics.append(MetricQC(name=metric_name, value = method))
            else:
                display_svg_group(
                    svg_list=svg_list,
                    sub_id=sub_id,
                    qc_name=qc_name,
                    metric_name=metric_name,
                    subject_metrics=run_metrics,
                    ses=ses,
                    task=task,
                    run=run
                )
        # --- Require rerun QC radio ---
        metric = "require_rerun"
        stored_val = get_val(f"sub-{sub_id}", ses, task, run, metric)
        # stored_val = None
        options = ("YES", "NO")
        require_rerun = st.radio(
            f"Require rerun?",
            options,
            key=f"{sub_id}_{ses}_{task}_{run}_rerun",
            index=options.index(stored_val) if stored_val in options else None,
        )
        if require_rerun is None:
            final_qc = None
        else:
            final_qc = "FAIL" if require_rerun == "YES" else "PASS"
        # Notes
        metric = "notes"
        stored_val = get_val(f"sub-{sub_id}", ses, task, run, metric)
        notes = st.text_input(f"***NOTES***", key=f"{sub_id}_{ses}_{task}_{run}_notes", value=stored_val)
        run_metrics.append(MetricQC(name="QC_notes", notes=notes))
        # all_metrics = global_metrics + run_metrics
    # st.write(subject_metrics)
    
    
    # Create QCRecord
        record = QCRecord(
            subject_id=sub_id,
            session_id=ses,
            task_id=task,
            run_id=run,
            pipeline="fmriprep-23.2.3",
            complete_timestamp= None,
            rater=rater_name,
            require_rerun=require_rerun,
            final_qc=final_qc,
            metrics=run_metrics,
        )
        qc_records.append(record)
    # st.session_state.current_batch_qc[sub_id] = record.model_dump()

# Pagination Controls - MOVED TO TOP
bottom_menu = st.columns((1, 2, 1))
# Update batch size first
with bottom_menu[2]:
    new_batch_size = st.selectbox(
        "Page Size",
        options=[1, 10, 20],
        index=(
            [1, 10, 20].index(st.session_state.batch_size)
            if st.session_state.batch_size in [1, 10, 20]
            else 0
        ),
    )

    # If batch size changed, reset to page 1
    if new_batch_size != st.session_state.batch_size:
        st.session_state.batch_size = new_batch_size
        st.session_state.current_page = 1
        st.rerun()

# Calculate total pages with current batch size
total_pages = max(1, math.ceil(total_rows / st.session_state.batch_size))
# Navigation controls
with bottom_menu[1]:
    col1, col2, col3 = st.columns([1, 1, 1], gap="small")

    if col1.button("⬅️"):
        if st.session_state.current_page > 1:
            st.session_state.current_page -= 1
            st.rerun()  # Force rerun to update immediately

    new_page = col2.number_input(
        "Page",
        min_value=1,
        max_value=total_pages,
        value=st.session_state.current_page,
        step=1,
    )

    # Update current page if changed
    if new_page != st.session_state.current_page:
        st.session_state.current_page = new_page
        st.rerun()

    if col3.button("➡️"):
        out_path = save_qc_results_to_csv(out_file, qc_records)
        if st.session_state.current_page < total_pages:
            st.session_state.current_page += 1
            st.rerun()  # Force rerun to update immediately

with bottom_menu[0]:
    st.markdown(f"Page **{st.session_state.current_page}** of **{total_pages}**")

# st.write("The current session state is:", qc_records)

st.write("The current session state is:", len(st.session_state))
# if st.button("Save QC results to CSV"):
#     out_path = save_qc_results_to_csv(out_file, qc_records)
#     st.success(f"QC results saved to: {out_path}")

st.markdown(
    """
    <a href="#top_of_page">
        <button style="
            background-color: white; 
            border: 1px solid #d1d5db; 
            padding: 0.5rem 1rem; 
            border-radius: 0.5rem; 
            cursor: pointer;">
            ⬆️ Scroll to Top
        </button>
    </a>
    """,
    unsafe_allow_html=True
)

# %%
