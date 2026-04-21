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
    coverage_qc_fragment,
    fd_censoring_fragment,
    qc_metric_fragment,
    render_pagination,
    render_run_tabs,
    render_scroll_to_top,
    render_session_tabs,
    verdict_fragment,
)

from utils import (
    collect_subject_qc,
    get_current_batch,
    get_metrics_from_csv,
    load_qc_configs,
    save_coverage_results_to_csv,
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
out_file          = Path(out_dir) / "XCPD_QC_status.csv"
coverage_out_dir = Path(out_dir) / "coverage_QC"
pipeline_version, qc_configs, fd_threshold = load_qc_configs("xcpd", config_file=qc_config)


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

    save_records   = []
    coverage_rows  = []
    for _, row in current_batch_pre.iterrows():
        sub_id = row["participant_id"].split("_")[0].split("-")[1]
        qc_bundles = _cached_collect_subject_qc(xcpd_dir, sub_id, tuple(qc_configs))

        for key in qc_bundles.keys():
            if key.label == "subject-level QC":
                continue
            expected_here = [c[2] for c in qc_configs if c[3] == "svg"]
            record = collect_qc_record(
                sub_id=sub_id,
                ses=key.ses,
                task=key.task,
                run=key.run,
                metric_ids=expected_here,
                qc_configs=qc_configs,
                rater_name=st.session_state.get("rater_name_input", ""),
                pipeline=pipeline_version,
                get_val=get_val,
            )
            save_records.append(record)

            # Collect coverage rows for the separate coverage CSV
            cov_key = f"{sub_id}_{key.ses}_{key.task}_{key.run}_coverage_results"
            coverage_rows.extend(st.session_state.get(cov_key, []))

    if save_records:
        save_qc_results_to_csv(out_file, save_records)
        _load_csv_data.clear()

    if coverage_rows:
        save_coverage_results_to_csv(coverage_out_dir, coverage_rows)

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

    # Group scan-level keys by (session, task)
    by_ses_task = defaultdict(list)
    for key in qc_bundles.keys():
        if key.label != "subject-level QC":
            by_ses_task[(key.ses, key.task)].append(key)

    ses_task_keys = sorted(by_ses_task.keys(), key=lambda x: (x[0] or "", x[1] or ""))

    def _fmt_ses_task(lbl):
        ses_l, task_l = lbl
        parts = [p for p in (ses_l, task_l) if p]
        return " | ".join(parts) if parts else "(no session)"

    outer_tabs, ses_task_keys = render_session_tabs(
        sub_id, ses_task_keys, fmt=_fmt_ses_task
    )
    # Lazy rendering: only render the active tab's content
    # _outer_tab_key match the key in render_session_tabs
    _outer_tab_key = f"session_tabs_{sub_id}"

    # Empty string on first load (nothing clicked yet).
    selected_label = st.session_state.get(_outer_tab_key, "")
    # find the (ses, task) tuple whose formatted label matches what was clicked
    _active_ses_task = None
    for stk in ses_task_keys:
        formatted = _fmt_ses_task(stk)      # e.g. "ses-01 | rest-01"
        if formatted == selected_label:
            _active_ses_task = stk          # e.g. ("ses-01", "rest-01")
            break
    # if nothing matched (first load, selected_label is ""), default to first
    if _active_ses_task is None:
        _active_ses_task = ses_task_keys[0] if ses_task_keys else None

    # _active_ses_task = next(
    #     (stk for stk in ses_task_keys if _fmt_ses_task(stk) == st.session_state.get(_outer_tab_key, "")),
    #     ses_task_keys[0] if ses_task_keys else None,
    # )

    for outer_tab, ses_task in zip(outer_tabs, ses_task_keys):
        with outer_tab:
            if ses_task != _active_ses_task:
                continue
            # Give the list of QCKeys for the active session
            run_keys = sorted(by_ses_task[ses_task], key=lambda k: k.run or "")
            run_tab_key = f"run_tabs_{sub_id}_{ses_task[0]}_{ses_task[1]}"
            run_pairs = render_run_tabs(run_keys, run_tab_key)

            # Lazy rendering: only render the active run tab
            # Step 1: what run label did the user click?
            selected_run_label = st.session_state.get(run_tab_key, "")

            # Step 2: find the QCKey whose run label matches
            _active_run_key = None
            for _, run_key in run_pairs:
                if (run_key.run or "(no run)") == selected_run_label:
                    _active_run_key = run_key
                    break

            # Step 3: default to first run on first load
            if _active_run_key is None:
                _active_run_key = run_pairs[0][1] if run_pairs else None
            # _active_run_key = next(
            #     (k for _, k in run_pairs if (k.run or "(no run)") == st.session_state.get(run_tab_key, "")),
            #     run_pairs[0][1] if run_pairs else None,
            # )

            for run_tab, key in run_pairs:
                with run_tab:
                    if len(run_pairs) > 1 and key != _active_run_key:
                        continue
                    ses, task, run = key.ses, key.task, key.run
                    expected_here = [c[2] for c in qc_configs if c[3] == "svg"]
                    found_map     = {item.metric_name: item for item in qc_bundles[key]}

                    with st.container():
                        # SVG/HTML metrics — rater radio buttons + missing warnings
                        for metric_id in expected_here:
                            if metric_id in found_map:
                                qc_metric_fragment(
                                    svg_list=found_map[metric_id].svg_list,
                                    sub_id=sub_id,
                                    qc_name=found_map[metric_id].qc_name,
                                    metric_name=metric_id,
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

                        # Non-SVG metrics — dispatch by file_type
                        for file_type in ("hdf5", "tsv"):
                            expected_non_svg = [c for c in qc_configs if c[3] == file_type]
                            for pattern, qc_name, metric_id, _ in expected_non_svg:
                                item = found_map.get(metric_id)
                                if item is None:
                                    st.warning(
                                        f"⚠️ **Missing file:** '{qc_name}' ({metric_id}) "
                                        f"[{file_type}] not found — check the pattern '{pattern}' in qc_config."
                                    )
                                    continue
                                # We no longer need to display the censoring curve so comment out the HDF5 fragment to speed up rendering. We can re-enable if we want to show it again in the future.
                                if file_type == "hdf5":
                                    # continue
                                    fd_censoring_fragment(
                                        sub_id=sub_id,
                                        h5_path=item.svg_list[0],
                                        # metric="remaining_minutes",
                                        threshold=fd_threshold,
                                    )
                                else:
                                    coverage_qc_fragment(
                                        sub_id=sub_id,
                                        tsv_list=item.svg_list,
                                        ses=ses,
                                        task=task,
                                        run=run,
                                    )

                        # Warn about any non-SVG items with an unexpected file_type
                        for item in qc_bundles[key]:
                            if item.file_type not in ("svg", "hdf5", "tsv"):
                                st.warning(f"⚠️ Unrecognized file type '{item.file_type}' for metric '{item.metric_name}'. Cannot render QC fragment.")

                        verdict_fragment(
                            sub_id=sub_id,
                            ses=ses,
                            task=task,
                            run=run,
                            get_val=get_val,
                        )

                        if st.button(
                            "💾 Save this tab",
                            key=f"save_tab_{sub_id}_{ses}_{task}_{run}",
                        ):
                            record = collect_qc_record(
                                sub_id=sub_id,
                                ses=ses,
                                task=task,
                                run=run,
                                metric_ids=expected_here,
                                qc_configs=qc_configs,
                                rater_name=st.session_state.get("rater_name_input", ""),
                                pipeline=pipeline_version,
                            )
                            save_qc_results_to_csv(out_file, [record])
                            _load_csv_data.clear()
                            cov_key = f"{sub_id}_{ses}_{task}_{run}_coverage_results"
                            cov_rows = st.session_state.get(cov_key, [])
                            if cov_rows:
                                save_coverage_results_to_csv(coverage_out_dir, cov_rows)
                            st.toast("Saved!", icon="✅")
            
# Pagination controls + scroll anchor

render_pagination(total_rows)
render_scroll_to_top()

# Debug info (remove in production)
st.write("Session state size:", len(st.session_state))