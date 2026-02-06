# %%
import math
import os
import pickle
import re
from argparse import ArgumentParser
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, NamedTuple
from dataclasses import dataclass
from functools import wraps
import numpy as np
import pandas as pd
import streamlit as st
from models import MetricQC, QCRecord
from bs4 import BeautifulSoup
from collections import defaultdict 

# Debug: show keys on every rerun
# st.write("Session state before initialized:", st.session_state)

st.markdown('<div id="top_of_page"></div>', unsafe_allow_html=True)

def parse_args(args=None):
    parser = ArgumentParser("fmriprep QC")

    parser.add_argument(
        "--qsiprep_dir",
        help=(
            "The root directory of fMRI preprocessing derivatives. "
            "For example, /SCanD_project/data/local/derivatives/qsiprep/0.22.0."
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
qsiprep_dir = args.qsiprep_dir
out_dir = args.out_dir

participants_df = pd.read_csv(participant_labels, delimiter="\t")

st.title("QSIPrep QC")
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

    rows = []
    for rec in qc_records:
        # 1. Start with IDs
        row = {
            "subject": f"sub-{rec.subject_id}" if not str(rec.subject_id).startswith("sub-") else rec.subject_id,
            "session": rec.session_id,
            "task": rec.task_id, 
            "run": rec.run_id,
            "pipeline": rec.pipeline,
            "complete_timestamp": rec.complete_timestamp,
        }

        # 2. Add Metrics (Dynamically inserted here in the middle)
        for m in rec.metrics:
            metric_name = m.name.lower().replace("-", "_")
            if m.qc is not None:
                row[metric_name] = m.qc
            if hasattr(m, 'value') and m.value is not None:
                row[f"{metric_name}_value"] = m.value

        # 3. Add Verdicts (Always at the end of the dict)
        row.update({
            "require_rerun": rec.require_rerun,
            "rater": rec.rater,
            "final_qc": rec.final_qc,
            "notes": next((m.notes for m in rec.metrics if m.name == "QC_notes"), None),
        })
        rows.append(row)

    df = pd.DataFrame(rows)
    
    if out_file.exists():
        # Load existing and ensure we align the new data to the OLD column order
        df_existing = pd.read_csv(out_file)
        
        # Combine them
        df = pd.concat([df_existing, df], ignore_index=True)
        
        # Deduplicate
        cols_to_fix = ["subject", "session", "task", "run"]
        for col in cols_to_fix:
            if col in df.columns:
                df[col] = df[col].replace({pd.NA: None, np.nan: None, "": None, "None": None})

        df = df.drop_duplicates(
            subset=["subject", "session", "task", "run", "pipeline"], 
            keep="last"
        )

        # FIX: Ensure any NEW columns (like sdc_qc) don't shove 'notes' out of the way
        # We move 'notes', 'final_qc', etc., back to the end one last time
    end_cols = ["require_rerun", "rater", "final_qc", "notes"]
    
    # Create list of columns: (All columns NOT in end_cols) + (The end_cols)
    cols = [c for c in df.columns if c not in end_cols] + [c for c in end_cols if c in df.columns]
    df = df[cols]

    # Sort so Global (NaN) is always first
    df = df.sort_values(by=["subject", "session", "task", "run"], na_position='first').reset_index(drop=True)
    df.to_csv(out_file, index=False, na_rep='')

    return out_file

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

def get_qc_key(filepath: Path) -> QCKey:
    """Extracts BIDS entities once and returns a structured QC-key."""
    fname = filepath.name
    
    ses = re.search(r"ses-([a-zA-Z0-9]+)", fname)
    task = re.search(r"task-([a-zA-Z0-9]+)", fname)
    run = re.search(r"run-([0-9]+)", fname)
    
    entities = {
        "ses": f"ses-{ses.group(1)}" if ses else None,
        "task": task.group(1) if task else None,
        "run": f"run-{run.group(1)}" if run else None
    }
    
    # Build label
    if entities["run"]:
        label = f"run-level QC"
    elif entities["task"]:
        label = f"task-level QC"
    elif entities["ses"]:
        label = f"session-level QC"
    else:
        label = "subject-level QC"
    
    return QCKey(label=label, **entities)

def collect_subject_qc(qsiprep_dir: Path, sub_id: str, configs: List) -> Dict[QCKey, List[QCEntry]]:
    
    qsiprep_dir = Path(qsiprep_dir)
    figures_path = qsiprep_dir / f"sub-{sub_id}" / "figures"
    
    search_paths = [qsiprep_dir, figures_path]

    temp_bundles = defaultdict(dict)

    for pattern, qc_name, metric_id in configs:
        for path in search_paths:
            if not path.exists():
                continue
            
            # Use a glob that ensures we only pick up files for THIS subject
            # This is important in the root dir where other subjects' HTMLs live
            search_pattern = f"sub-{sub_id}*{pattern}*"
            
            for f in path.glob(search_pattern):
                if f.suffix not in {".svg", ".html"}:
                    continue
                
                key = get_qc_key(f)
                
                if metric_id not in temp_bundles[key]:
                    temp_bundles[key][metric_id] = QCEntry(
                        svg_list=[f], 
                        qc_name=qc_name, 
                        metric_name=metric_id
                    )
                else:
                    # Avoid adding the same file twice if paths overlap
                    if f not in temp_bundles[key][metric_id].svg_list:
                        temp_bundles[key][metric_id].svg_list.append(f)

    return {key: list(metrics.values()) for key, metrics in temp_bundles.items()}


def extract_fieldmap_method(html_path):
    """Parse a single HTML file and extract the susceptibility distortion correction method."""

    results = {}

    with open(html_path, "r") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    # Restrict to Diffusion section
    diffusion_div = soup.find("div", id="Diffusion")

    session_divs = diffusion_div.find_all(
        "div", id=lambda x: x and x.startswith("ses-")
    )
    if session_divs:
        targets = [(ses_div["id"], ses_div) for ses_div in session_divs]
    else:
        targets = [("nosession", diffusion_div)]

    for session_id, container in targets:
        # session_id = ses_div["id"]

        h3 = container.find(
            "h3",
            class_="elem-title",
            string=lambda s: s and s.startswith("Susceptibility distortion correction")
        )

        if h3:
            # Extract text inside parentheses
            match = re.search(r"\((.*?)\)", h3.text)
            method = match.group(1) if match else "UNKNOWN"
        else:
            method = "NOT FOUND"

        results[session_id] = method
    return method

total_rows, current_batch = get_current_batch(
    participants_df, st.session_state.current_page, st.session_state.batch_size
)

# Save to CSV
# out_dir = "/projects/ttan/tigrbid-QC/outputs"
now = datetime.now()
# timestamp = now.strftime("%Y%m%d")  # e.g., 20250917
out_file = Path(out_dir) / f"QSIPrep_QC_status.csv"

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

# 1. Configuration for the QC metrics you want to collect

qc_configs = [
    ("seg_brainmask", "T1 tissue segmentation", "segmentation_qc"),
    ("desc-sdc_b0", "Susceptible Distortion Correction", "sdc_qc"),
    ("coreg", "B0 to T1 Coregistration", "b0_2_t1_qc"),
    ("t1_2_mni", "T1 to MNI Coregistration", "t1_2_mni_qc"),
    (".html", "FieldMap Method", "fieldmap_method")
]

for _, row in current_batch.iterrows():
    subj = row["participant_id"]
    parts = subj.split("_")
    sub_id = parts[0].split("-")[1]

    qc_bundles = collect_subject_qc(Path(qsiprep_dir), sub_id, qc_configs)
    # sorted_keys = sorted(qc_bundles.keys(), key=lambda x: (x.label != "Global_Subject_Level", x.label))
    
    # 1. Temporary storage to keep bundle metrics separate before saving
    temp_bundles_to_save = []

    for key in qc_bundles.keys():
        bundle_metrics = []
        ses, task, run = key.ses, key.task, key.run

        for item in qc_bundles[key]:
            if item.metric_name == "fieldmap_method":
                for path in item.svg_list:
                    method = extract_fieldmap_method(path)
                    bundle_metrics.append(MetricQC(name=item.metric_name, value=method))
            else:
                display_svg_group(
                    svg_list=item.svg_list,
                    sub_id=sub_id,
                    qc_name=item.qc_name,
                    metric_name=item.metric_name,
                    subject_metrics=bundle_metrics,
                    ses=ses, task=task, run=run
                )
        
        # Save this bundle's data for Pass 2
        temp_bundles_to_save.append({
            "ses": ses, "task": task, "run": run, "metrics": bundle_metrics
        })

    # --- 2. Master Verdict: OUTSIDE the sorted_keys loop ---
    st.divider()
    # st.subheader("Final Subject Verdict")

    # Look up existing values at the Subject Level (using None for entities)
    stored_rerun = get_val(f"sub-{sub_id}", metric="require_rerun")
    stored_notes = get_val(f"sub-{sub_id}", metric="notes")
    
    options = ("YES", "NO")
    # Note: Unique key using ONLY sub_id to ensure only one radio button exists
    require_rerun = st.radio(
        "Require rerun?",
        options,
        key=f"{sub_id}_final_rerun_choice",
        index=options.index(stored_rerun) if stored_rerun in options else None,
        horizontal=True
    )

    final_qc = "FAIL" if require_rerun == "YES" else ("PASS" if require_rerun == "NO" else None)
    notes = st.text_input(f"***NOTES***", key=f"{sub_id}_final_notes_input", value=stored_notes if stored_notes else "")

    # --- 3. Pass 2: Create QCRecords for every bundle found ---
    for entry in temp_bundles_to_save:
        # Attach the shared notes to every record
        entry["metrics"].append(MetricQC(name="QC_notes", notes=notes))
        
        record = QCRecord(
            subject_id=sub_id,
            session_id=entry["ses"],
            task_id=entry["task"],
            run_id=entry["run"],
            pipeline="qsiprep-0.22.0",
            complete_timestamp=None,
            rater=rater_name,
            require_rerun=require_rerun,
            final_qc=final_qc,
            metrics=entry["metrics"],
        )
        qc_records.append(record)

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
