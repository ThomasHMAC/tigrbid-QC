"""
xcpd_qc.py
----------
Main entry point for the XCPD QC Streamlit app.

Run with:
    streamlit run xcpd_qc.py -- \
        --xcpd_dir /path/to/xcpd \
        --participant_labels /path/to/participants.tsv \
        --output_dir /path/to/outputs \
        --qc_config /path/to/xcpd_qc.json
"""

import math
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from ui import (
    collect_qc_record,
    qc_metric_fragment,
    render_pagination,
    render_scroll_to_top,
    verdict_fragment,
)
from utils import (
    collect_subject_qc,
    get_current_batch,
    get_metrics_from_csv,
    load_qc_configs,
    save_qc_results_to_csv,
)


# Page config
st.set_page_config(layout="wide")
st.markdown('<div id="top_of_page"></div>', unsafe_allow_html=True)

def parse_args(args=None):
    parser = ArgumentParser("fmriprep QC")

    parser.add_argument(
        "--xcpd_dir",
        help=(
            "The root directory of xcp preprocessing derivatives. "
            "For example, /SCanD_project/data/local/derivatives/xcpd/0.7.3."
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
    parser.add_argument(
        "--qc_config",
        dest="qc_config",
        help="Path to the QC configuration file",
        required=True,
    )

    return parser.parse_args(args)


args = parse_args()

xcpd_dir          = args.xcpd_dir
participant_labels = args.participant_labels
out_dir           = args.out_dir
qc_config         = args.qc_config

participants_df = pd.read_csv(participant_labels, delimiter="\t")
out_file        = Path(out_dir) / "XCPD_QC_status.csv"
qc_configs      = load_qc_configs("xcpd", config_file=qc_config)


# Session state initialisation
def init_session_state():
    defaults = {
        "current_page":      1,
        "batch_size":        1,
        "current_batch_qc": {},
        "_save_and_advance": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()

# Load existing CSV once per session (cached so it doesn't re-read on every
# fragment interaction)

@st.cache_data(show_spinner=False)
def _load_csv_data(path: str) -> dict:
    """Cache only data_dict (picklable). get_val is a closure so can't be cached."""
    data_dict, _ = get_metrics_from_csv(Path(path))
    return data_dict


@st.cache_data(show_spinner=False)
def _cached_collect_subject_qc(xcpd_dir: str, sub_id: str, qc_configs: tuple):
    """Cache filesystem globbing so it doesn't repeat on every rerun."""
    return collect_subject_qc(Path(xcpd_dir), sub_id, list(qc_configs))


data_dict = _load_csv_data(str(out_file))


def get_val(sub_id, ses_id=None, task_id=None, run_id=None, metric=None):
    """Reconstruct each rerun from the cached data_dict — cheap, no disk I/O."""
    if sub_id is None or metric is None:
        return None
    s_sub = sub_id if sub_id.startswith("sub-") else f"sub-{sub_id}"
    val = data_dict.get((s_sub, ses_id, task_id, run_id), {}).get(metric)
    if val is not None and not (isinstance(val, float) and np.isnan(val)):
        return val
    return None


# Handle save-and-advance flag (set by ➡️ button in pagination)
if st.session_state.get("_save_and_advance"):
    st.session_state["_save_and_advance"] = False

    # Re-collect records from current session state and save
    total_rows_pre, current_batch_pre = get_current_batch(
        participants_df,
        st.session_state.current_page,
        st.session_state.batch_size,
    )

    save_records = []
    for _, row in current_batch_pre.iterrows():
        sub_id = row["participant_id"].split("_")[0].split("-")[1]
        qc_bundles = _cached_collect_subject_qc(xcpd_dir, sub_id, tuple(qc_configs))

        for key in qc_bundles.keys():
            if key.label == "subject-level QC":
                continue
            expected_here = ["IQM_qc", "ESQC_bold_qc", "connectivity_matrix_qc"]
            record = collect_qc_record(
                sub_id=sub_id,
                ses=key.ses,
                task=key.task,
                run=key.run,
                metric_ids=expected_here,
                qc_configs=qc_configs,
                rater_name=st.session_state.get("rater_name_input", ""),
            )
            save_records.append(record)

    if save_records:
        save_qc_results_to_csv(out_file, save_records)
        # Invalidate CSV cache so next page sees fresh data
        # _load_csv.clear()

    if st.session_state.current_page < math.ceil(
        len(participants_df) / st.session_state.batch_size
    ):
        st.session_state.current_page += 1

    st.rerun()

st.title("XCPD QC")
rater_name = st.text_input("Rater name:", key="rater_name_input")
st.write("You entered:", rater_name)

total_rows, current_batch = get_current_batch(
    participants_df, st.session_state.current_page, st.session_state.batch_size
)

# UI
for _, row in current_batch.iterrows():
    subj   = row["participant_id"]
    sub_id = subj.split("_")[0].split("-")[1]

    qc_bundles = _cached_collect_subject_qc(xcpd_dir, sub_id, tuple(qc_configs))

    if not qc_bundles:
        st.warning(f"No QC images found for sub-{sub_id}")
        continue

    # Group scan-level keys by session
    by_session = defaultdict(list)
    for key in qc_bundles.keys():
        if key.label != "subject-level QC":
            by_session[key.ses].append(key)

    session_labels = sorted(by_session.keys(), key=lambda x: x or "")
    tabs = st.tabs(session_labels)

    for tab, ses_label in zip(tabs, session_labels):
        with tab:
            for key in by_session[ses_label]:
                ses, task, run = key.ses, key.task, key.run
                expected_here = ["IQM_qc", "ESQC_bold_qc", "connectivity_matrix_qc"]
                found_map     = {item.metric_name: item for item in qc_bundles[key]}

                with st.container():
                    for metric_id in expected_here:
                        if metric_id in found_map:
                            item = found_map[metric_id]
                            qc_metric_fragment(
                                svg_list=item.svg_list,
                                sub_id=sub_id,
                                qc_name=item.qc_name,
                                metric_name=item.metric_name,
                                get_val=get_val,
                                ses=ses,
                                task=task,
                                run=run,
                            )
                        else:
                            pretty_label = next(
                                (c[1] for c in qc_configs if c[2] == metric_id), metric_id
                            )
                            st.warning(f"⚠️ **Missing Image:** '{pretty_label}' ({metric_id}) not found.")

                    verdict_fragment(
                        sub_id=sub_id,
                        ses=ses,
                        task=task,
                        run=run,
                        get_val=get_val,
                    )
            
# Pagination controls + scroll anchor

render_pagination(total_rows)
render_scroll_to_top()

# Debug info (remove in production)
st.write("Session state size:", len(st.session_state))