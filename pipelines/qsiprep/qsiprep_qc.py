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
participant_labels = args.participant_labels
qsiprep_dir = args.qsiprep_dir
out_dir = args.out_dir

participants_df = pd.read_csv(participant_labels, delimiter="\t")

st.title("QSIPrep QC")
rater_name = st.text_input("Rater name:")
st.write("You entered:", rater_name)


def init_session_state():
    defaults = {
        "current_page": 1,
        "batch_size": 1,
        "current_batch_qc": {},
    }
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
    """
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for rec in qc_records:
        acq_value = next(
            (m.value for m in getattr(rec, "metrics", []) if getattr(m, "name", None) == "acq"),
            None,
        )

        row = {
            "subject": f"sub-{rec.subject_id}" if not str(rec.subject_id).startswith("sub-") else rec.subject_id,
            "session": rec.session_id,
            "task": rec.task_id,
            "acq": acq_value,
            "run": rec.run_id,
            "pipeline": rec.pipeline,
            "complete_timestamp": rec.complete_timestamp,
        }

        for m in rec.metrics:
            metric_name = m.name.lower().replace("-", "_")
            if m.qc is not None:
                row[metric_name] = m.qc
            if hasattr(m, "value") and m.value is not None:
                row[f"{metric_name}"] = m.value

        row.update(
            {
                "require_rerun": rec.require_rerun,
                "rater": rec.rater,
                "final_qc": rec.final_qc,
                "notes": next((m.notes for m in rec.metrics if m.name == "QC_notes"), None),
            }
        )
        rows.append(row)

    df = pd.DataFrame(rows)

    if out_file.exists():
        df_existing = pd.read_csv(out_file)
        df = pd.concat([df_existing, df], ignore_index=True)

        cols_to_fix = ["subject", "session", "task", "acq", "run"]
        for col in cols_to_fix:
            if col in df.columns:
                df[col] = df[col].replace({pd.NA: None, np.nan: None, "": None, "None": None})

        df = df.drop_duplicates(subset=["subject", "session", "task", "acq", "run", "pipeline"], keep="last")

    end_cols = ["require_rerun", "rater", "final_qc", "notes"]
    cols = [c for c in df.columns if c not in end_cols] + [c for c in end_cols if c in df.columns]
    df = df[cols]

    sort_cols = [c for c in ["subject", "session", "task", "acq", "run"] if c in df.columns]
    df = df.sort_values(by=sort_cols, na_position="first").reset_index(drop=True)
    df.to_csv(out_file, index=False, na_rep="")

    return out_file


# -----------------------
# ADD acq support here
# -----------------------
class QCKey(NamedTuple):
    label: str
    ses: Optional[str] = None
    task: Optional[str] = None
    acq: Optional[str] = None
    run: Optional[str] = None


@dataclass
class QCEntry:
    svg_list: List[Path]
    qc_name: str
    metric_name: str


def get_qc_key(filepath: Path) -> QCKey:
    """Extract BIDS entities once and return a structured QC-key (includes acq)."""
    fname = filepath.name

    ses = re.search(r"ses-([a-zA-Z0-9]+)", fname)
    task = re.search(r"task-([a-zA-Z0-9]+)", fname)
    acq = re.search(r"acq-([a-zA-Z0-9]+)", fname)
    run = re.search(r"run-([0-9]+)", fname)

    entities = {
        "ses": f"ses-{ses.group(1)}" if ses else None,
        "task": task.group(1) if task else None,
        "acq": f"acq-{acq.group(1)}" if acq else None,
        "run": f"run-{run.group(1)}" if run else None,
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


def collect_subject_qc(qsiprep_dir: Path, sub_id: str, configs: List) -> Dict[QCKey, List[QCEntry]]:
    qsiprep_dir = Path(qsiprep_dir)
    figures_path = qsiprep_dir / f"sub-{sub_id}" / "figures"

    search_paths = [qsiprep_dir, figures_path]
    temp_bundles = defaultdict(dict)

    for pattern, qc_name, metric_id in configs:
        for path in search_paths:
            if not path.exists():
                continue

            search_pattern = f"sub-{sub_id}*{pattern}*"

            for f in sorted(path.glob(search_pattern)):
                if f.suffix not in {".svg", ".html"}:
                    continue

                key = get_qc_key(f)

                if metric_id not in temp_bundles[key]:
                    temp_bundles[key][metric_id] = QCEntry(svg_list=[f], qc_name=qc_name, metric_name=metric_id)
                else:
                    if f not in temp_bundles[key][metric_id].svg_list:
                        temp_bundles[key][metric_id].svg_list.append(f)

    return {key: list(metrics.values()) for key, metrics in temp_bundles.items()}


def extract_fieldmap_method(html_path):
    """Parse a single HTML file and extract the susceptibility distortion correction method."""
    results = {}

    html_path = Path(html_path)
    if not html_path.exists():
        return {"nosession": "HTML_NOT_FOUND"}

    with open(html_path, "r") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    diffusion_div = soup.find("div", id="Diffusion")
    if diffusion_div is None:
        return {"nosession": "DIFFUSION_SECTION_NOT_FOUND"}

    session_divs = diffusion_div.find_all("div", id=lambda x: x and x.startswith("ses-"))
    if session_divs:
        targets = [(ses_div["id"], ses_div) for ses_div in session_divs]
    else:
        targets = [("nosession", diffusion_div)]

    for session_id, container in targets:
        method = "NOT FOUND"
        sdc_li = container.find("li", string=lambda s: s and "Susceptibility distortion correction" in s)
        if sdc_li:
            text_content = sdc_li.get_text()
            if ":" in text_content:
                method = text_content.split(":", 1)[1].strip()

        ses_key = session_id.split("_", 1)[0] if isinstance(session_id, str) else session_id
        results[ses_key] = method

    return results


total_rows, current_batch = get_current_batch(participants_df, st.session_state.current_page, st.session_state.batch_size)

now = datetime.now()
out_file = Path(out_dir) / "QSIPrep_QC_status.csv"


def get_metrics_from_csv(qc_results: Path):
    if not qc_results.exists():
        return {}, lambda *args, **kwargs: None

    df = pd.read_csv(qc_results)

    for col in ["session", "task", "acq", "run"]:
        if col not in df.columns:
            df[col] = None

    def clean_id(val, prefix):
        if pd.isna(val) or val == "":
            return None
        val_str = str(val)
        return val_str if val_str.startswith(prefix) else f"{prefix}{val_str}"

    data_dict = {}

    for _, row in df.iterrows():
        sub = clean_id(row.get("subject"), "sub-")
        ses = clean_id(row.get("session"), "ses-")
        task = row.get("task")
        acq = clean_id(row.get("acq"), "acq-")
        run = clean_id(row.get("run"), "run-")
        if pd.isna(task):
            task = None

        key = (sub, ses, task, acq, run)

        metrics = row.drop(
            ["subject", "session", "task", "acq", "run", "pipeline", "complete_timestamp", "rater"], errors="ignore"
        ).to_dict()
        data_dict[key] = metrics

    def get_val(sub_id, ses_id=None, task_id=None, acq_id=None, run_id=None, metric=None):
        if sub_id is None or metric is None:
            return None

        s_sub = sub_id if sub_id.startswith("sub-") else f"sub-{sub_id}"
        target_key = (s_sub, ses_id, task_id, acq_id, run_id)

        val = data_dict.get(target_key, {}).get(metric)
        if val is not None and not pd.isna(val):
            return val
        return None

    return data_dict, get_val


data_dict, get_val = get_metrics_from_csv(out_file)


def display_svg_group(
    svg_list: list[Path],
    sub_id: str,
    qc_name: str,
    metric_name: str,
    subject_metrics: list,
    ses=None,
    task=None,
    acq=None,
    run=None,
):
    """
    Display one or more SVGs with a QC radio button.
    ses/task/acq/run are optional; included to make Streamlit keys unique.
    """
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

        # Build unique Streamlit key (includes acq)
        key = f"{sub_id}"
        if ses:
            key += f"_{ses}"
        if task:
            key += f"_{task}"
        if acq:
            key += f"_{acq}"
        if run:
            key += f"_{run}"
        key += f"_{metric_name}"

        stored_val = get_val(
            sub_id=f"sub-{sub_id}",
            ses_id=ses,
            task_id=task,
            acq_id=acq,
            run_id=run,
            metric=metric_name,
        )

        qc_choice = st.radio(
            f"{qc_name} QC:",
            options,
            key=key,
            label_visibility="collapsed",
            index=options.index(stored_val) if stored_val in options else None,
        )

        subject_metrics.append(MetricQC(name=metric_name, qc=qc_choice))


qc_records = []

qc_configs = [
    ("seg_brainmask", "T1 tissue segmentation", "segmentation_qc"),
    ("desc-sdc_b0", "Susceptible Distortion Correction", "sdc_qc"),
    ("coreg", "B0 to T1 Coregistration", "b0_2_t1_qc"),
    ("t1_2_mni", "T1 to MNI Coregistration", "t1_2_mni_qc"),
]

for _, row in current_batch.iterrows():
    subj = row["participant_id"]
    parts = subj.split("_")
    sub_id = parts[0].split("-")[1]

    sub_root = Path(qsiprep_dir) / f"sub-{sub_id}"
    html_path = Path(qsiprep_dir) / f"sub-{sub_id}.html"
    if (not sub_root.exists()) and (not html_path.exists()):
        st.warning(f"⚠️ sub-{sub_id} is in participants.tsv but not found in QSIPrep outputs. Skipping.")
        continue

    fieldmap_methods = extract_fieldmap_method(html_path)
    qc_bundles = collect_subject_qc(Path(qsiprep_dir), sub_id, qc_configs)

    if not qc_bundles:
        st.warning(f"⚠️ No QC figures found for sub-{sub_id}. Skipping.")
        continue

    for key in qc_bundles.keys():
        bundle_metrics = []
        ses, task, acq, run = key.ses, key.task, key.acq, key.run

        bundle_metrics.append(MetricQC(name="acq", value=acq))

        if key.label == "subject-level QC":
            expected_here = ["segmentation_qc", "t1_2_mni_qc"]
        else:
            expected_here = ["sdc_qc", "b0_2_t1_qc"]

        found_map = {item.metric_name: item for item in qc_bundles[key]}

        for metric_id in expected_here:
            if metric_id in found_map:
                item = found_map[metric_id]

                if item.metric_name == "sdc_qc":
                    method_value = fieldmap_methods.get(ses) or fieldmap_methods.get("nosession", "UNKNOWN")
                    bundle_metrics.append(MetricQC(name="fieldmap_method", value=method_value))

                display_svg_group(
                    svg_list=item.svg_list,
                    sub_id=sub_id,
                    qc_name=item.qc_name,
                    metric_name=item.metric_name,
                    subject_metrics=bundle_metrics,
                    ses=ses,
                    task=task,
                    acq=acq,
                    run=run,
                )
            else:
                pretty_label = next((c[1] for c in qc_configs if c[2] == metric_id), metric_id)
                st.warning(f"⚠️ **Missing Image:** '{pretty_label}' ({metric_id}) was not found.")

        st.divider()

        stored_rerun = get_val(f"sub-{sub_id}", ses, task, acq, run, metric="require_rerun")
        stored_notes = get_val(f"sub-{sub_id}", ses, task, acq, run, metric="notes")

        options = ("YES", "NO")

        require_rerun = st.radio(
            "Require rerun?",
            options,
            key=f"{sub_id}_{ses}_{task}_{acq}_{run}_rerun",
            index=options.index(stored_rerun) if stored_rerun in options else None,
            horizontal=True,
        )

        final_qc = "FAIL" if require_rerun == "YES" else ("PASS" if require_rerun == "NO" else None)
        notes = st.text_input(
            f"***NOTES***",
            key=f"{sub_id}_{ses}_{task}_{acq}_{run}_notes",
            value=stored_notes if stored_notes else "",
        )
        bundle_metrics.append(MetricQC(name="QC_notes", notes=notes))

        record = QCRecord(
            subject_id=sub_id,
            session_id=ses,
            task_id=task,
            run_id=run,
            pipeline="qsiprep-0.22.0",
            complete_timestamp=None,
            rater=rater_name,
            require_rerun=require_rerun,
            final_qc=final_qc,
            metrics=bundle_metrics,
        )
        qc_records.append(record)

bottom_menu = st.columns((1, 2, 1))

with bottom_menu[2]:
    new_batch_size = st.selectbox(
        "Page Size",
        options=[1, 10, 20],
        index=([1, 10, 20].index(st.session_state.batch_size) if st.session_state.batch_size in [1, 10, 20] else 0),
    )

    if new_batch_size != st.session_state.batch_size:
        st.session_state.batch_size = new_batch_size
        st.session_state.current_page = 1
        st.rerun()

total_pages = max(1, math.ceil(total_rows / st.session_state.batch_size))

with bottom_menu[1]:
    col1, col2, col3 = st.columns([1, 1, 1], gap="small")

    if col1.button("⬅️"):
        out_path = save_qc_results_to_csv(out_file, qc_records)
        if st.session_state.current_page > 1:
            st.session_state.current_page -= 1
            st.rerun()

    new_page = col2.number_input(
        "Page",
        min_value=1,
        max_value=total_pages,
        value=st.session_state.current_page,
        step=1,
    )

    if new_page != st.session_state.current_page:
        st.session_state.current_page = new_page
        st.rerun()

    if col3.button("➡️"):
        out_path = save_qc_results_to_csv(out_file, qc_records)
        if st.session_state.current_page < total_pages:
            st.session_state.current_page += 1
            st.rerun()

with bottom_menu[0]:
    st.markdown(f"Page **{st.session_state.current_page}** of **{total_pages}**")

st.write("The current session state is:", len(st.session_state))

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
    unsafe_allow_html=True,
)

# %%
