"""
ui.py
-----
Streamlit UI components for the XCPD QC app.

Key design: each QC bundle is wrapped in @st.fragment so that selecting a
radio button only re-runs that fragment, NOT the entire page.
"""

import base64
import re
from pathlib import Path
from typing import Callable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from models import MetricQC, QCRecord
from utils import (
    QCEntry,
    QCKey,
    compute_coverage_qc,
    load_coverage,
    extract_fd_metrics_at_threshold,
    load_fd_curves,
)


def format_qc_label(option: str) -> str:
    mapping = {
        "PASS":      "**:green[PASS]**",
        "FAIL":      "**:red[FAIL]**",
        "UNCERTAIN": "**:yellow[UNCERTAIN]**",
        "YES":       "**:red[YES]**",
        "NO":        "**:green[NO]**"
    }
    return mapping.get(option, option)


@st.cache_data(show_spinner=False)
def _read_file(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


@st.cache_data(show_spinner=False)
def _svg_as_img_tag(path: str, max_width: int = 600) -> str:
    """
    Encode SVG as a base64 <img> tag.
    The browser treats it as an opaque image (no SVG DOM) — much faster to
    scroll and paint than inlined SVG XML.
    """
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return (
        f'<img src="data:image/svg+xml;base64,{b64}" '
        f'style="max-width:{max_width}px; width:100%; height:auto; display:block; margin:auto;">'
    )


# Single SVG/HTML group — wrapped in a fragment to prevent full re-renders

@st.fragment
def qc_metric_fragment(
    svg_list: List[Path],
    sub_id: str,
    qc_name: str,
    metric_name: str,
    get_val: Callable,
    ses: Optional[str] = None,
    task: Optional[str] = None,
    run: Optional[str] = None,
):
    """
    Renders one QC metric block (images + radio).
    Wrapped in @st.fragment so radio interaction doesn't trigger a full rerun.

    Returns nothing — writes the chosen value into st.session_state directly
    so the outer scope can read it when the user navigates / saves.
    """
    # Title
    parts = [f"sub-{sub_id}"]
    if task: parts.append(task)
    if ses:  parts.append(ses)
    if run:  parts.append(run)
    parts.append(f"- {qc_name} QC")
    sub_title = " ".join(parts)

    st.markdown(
        f"<h4 style='font-family:Arial; color:darkblue;'>{sub_title}</h4>",
        unsafe_allow_html=True,
    )

    # Displaying
    # with st.expander(qc_name, expanded=True):
    cols = st.columns(len(svg_list), vertical_alignment="center")
    for col, svg_path in zip(cols, svg_list):
        if not svg_path.exists():
            col.warning(f"Missing file: {svg_path.name}")
            continue

        name = svg_path.name
        if "preprocESQC_bold" in name:
            label = "Pre Regression"
        elif "postprocESQC_bold" in name:
            label = "Post Regression"
        else:
            label = ""

        with col:
            if svg_path.suffix == ".html":
                st.html(
                    f"<div style='max-height:400px; overflow:auto;'>"
                    f"{_read_file(str(svg_path))}</div>"
                )
            else:
                if label:
                    st.markdown(
                        f"<div style='text-align:center;font-size:18px;'><b>{label}</b></div>",
                        unsafe_allow_html=True,
                    )
                st.markdown(_svg_as_img_tag(str(svg_path), max_width=900), unsafe_allow_html=True)

    # Radio button
    # Build a unique key for this widget
    key_parts = [sub_id]
    if ses:  key_parts.append(ses)
    if task: key_parts.append(task)
    if run:  key_parts.append(run)
    key_parts.append(metric_name)
    widget_key = "_".join(key_parts)

    raw_options = ("PASS", "FAIL", "UNCERTAIN")

    # Resolve initial value: session_state wins over CSV
    stored_val = get_val(
        sub_id=f"sub-{sub_id}",
        ses_id=ses,
        task_id=task,
        run_id=run,
        metric=metric_name,
    )

    with st.container(border=True):
        st.radio(
            f"{qc_name} QC:",
            options=raw_options,
            key=widget_key,
            label_visibility="collapsed",
            format_func=format_qc_label,
            index=raw_options.index(stored_val) if stored_val in raw_options else None,
        )

    st.divider()

# Verdict block (require_rerun + notes) — also a fragment
@st.fragment
def verdict_fragment(
    sub_id: str,
    ses: Optional[str],
    task: Optional[str],
    run: Optional[str],
    get_val: Callable,
):
    """
    Renders the 'Require rerun?' radio and the notes text input.
    Wrapped in @st.fragment for the same isolation reason.
    """
    stored_rerun = st.session_state.get(f"{sub_id}_{ses}_{task}_{run}_rerun") or \
                   get_val(f"sub-{sub_id}", ses, task, run, metric="require_rerun")
    stored_notes = st.session_state.get(f"{sub_id}_{ses}_{task}_{run}_notes") or \
                   get_val(f"sub-{sub_id}", ses, task, run, metric="notes")

    options = ("YES", "NO")
    st.radio(
        "**Require rerun**?",
        options,
        key=f"{sub_id}_{ses}_{task}_{run}_rerun",
        index=options.index(stored_rerun) if stored_rerun in options else None,
        horizontal=True,
        format_func=format_qc_label
    )
    st.text_input(
        "***NOTES***",
        key=f"{sub_id}_{ses}_{task}_{run}_notes",
        value=stored_notes if stored_notes else "",
    )


@st.fragment
def fd_censoring_fragment(
    sub_id: str,
    h5_path: Path,
    # metric: str = "ratios",
    threshold: float = 1,
):
    """
    Renders the FD threshold censoring curve for one functional scan.
    metric: one of 'ratios', 'remaining_frames', 'remaining_minutes'
    """
    # ylabel_map = {
    #     "ratios":            ("Remaining / Total Frames", 0.8),
    #     "remaining_frames":  ("Remaining Frames",         None),
    #     "remaining_minutes": ("Remaining Time (min)",     "50%"),
    # }
    # ylabel, hline = ylabel_map[metric]

    try:
        data = load_fd_curves(h5_path)
    except Exception as e:
        st.warning(f"Could not load FD curves from `{h5_path.name}`: {e}")
        return

    # Resolve the 50% sentinel after data is available
    # if hline == "50%":
    #     max_val = data[metric].max() if hasattr(data[metric], "max") else max(data[metric])
    #     hline = max_val * 0.5

    label_parts = [p for p in [
        f"ses-{data['session']}" if data["session"] else None,
        f"task-{data['task']}"   if data["task"]    else None,
        f"run-{data['run']}"     if data["run"]      else None,
    ] if p]
    curve_label = "_".join(label_parts) if label_parts else h5_path.stem
    # fig, ax = plt.subplots(figsize=(5, 4))
    # ax.plot(data["fds"], data[metric], alpha=0.7, label=curve_label)
    # if hline is not None:
    #     ax.axhline(y=hline, linestyle="--", color="red")
    # ax.set_xlim(0, 0.5)
    # # ax.set_title(f"sub-{sub_id} — FD Censoring Curve", fontsize=10, loc='left')
    # ax.set_xlabel("FD Threshold", fontsize=8)
    # ax.set_ylabel(ylabel, fontsize=8)
    # ax.grid(True)
    # ax.legend(fontsize=6, loc="lower right")
    # fig.tight_layout()

    # st.markdown(
    #     f"<h4 style='font-family:Arial; color:darkblue;'>sub-{sub_id} — FD Censoring Curve</h4>",
    #     unsafe_allow_html=True,
    # )
    # _, center, _ = st.columns([1, 2, 1])
    # with center:
    #     st.pyplot(fig, width='content')
        
    # plt.close(fig)

    fd_metrics = extract_fd_metrics_at_threshold(h5_path)
    st.markdown(
    f"<h4 style='font-family:Arial; color:darkblue;'>sub-{sub_id} {curve_label} — FD Censoring Curve</h4>",
    unsafe_allow_html=True,
    )
    fd_metrics = extract_fd_metrics_at_threshold(h5_path,threshold)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("FD Threshold", fd_metrics["fd_threshold"])
    col2.metric("Total (min)", fd_metrics["total_minutes"])
    col3.metric("Remaining (min)", fd_metrics["remaining_minutes"])
    col4.metric("Min Duration QC", fd_metrics["min_duration_qc"])
    if fd_metrics:
        state_key = f"{sub_id}_ses-{data['session']}_{data['task']}_run-{data['run']}_fd_metrics"
        st.session_state[state_key] = fd_metrics

    st.divider()


@st.fragment
def coverage_qc_fragment(
    sub_id: str,
    tsv_list: List[Path],
    ses: Optional[str],
    task: Optional[str],
    run: Optional[str],
    threshold: float = 0.5,
    fail_pct_cutoff: float = 10,
):
    """
    Receives coverage TSV paths from collect_subject_qc, computes QC per atlas,
    plots failing atlases, and stores results in st.session_state for later saving.
    """
    if not tsv_list:
        return

    st.markdown(
        "<h4 style='font-family:Arial; color:darkblue;'>Atlas Coverage QC</h4>",
        unsafe_allow_html=True,
    )

    qc_results = []
    num_atlas_failed = []
    for tsv_file in tsv_list:
        result = load_coverage(tsv_file)
        if result is None:
            continue
        atlas, parcels = result
        qc = compute_coverage_qc(
            parcels,
            subject=f"sub-{sub_id}",
            session=ses,
            task=task,
            run=run,
            atlas=atlas,
            threshold=threshold,
            fail_pct_cutoff=fail_pct_cutoff,
        )
        
        if qc['coverage_QC'] != "PASS":
           num_atlas_failed.append(atlas)

        qc_results.append(qc)

        # display_atlas = atlas
        # status_color = "green" if qc["coverage_QC"] == "PASS" else "red"
        # st.markdown(
        #     f"**:{status_color}[{qc['coverage_QC']}]** — `{display_atlas}` | "
        #     f"{qc['n_failed_parcels']}/{qc['n_parcels']} parcels failed "
        #     f"({qc['pct_failed']}%)"
        # )

        if qc["coverage_QC"] == "FAIL":
            parcel_names  = list(parcels.index)
            parcel_values = parcels.values.astype(float)
            failed_mask   = parcel_values < threshold
            failed_names  = [n for n, m in zip(parcel_names, failed_mask) if m]
            failed_values = parcel_values[failed_mask]

            fig, ax = plt.subplots(figsize=(7, max(3, len(failed_names) * 0.3)))
            ax.barh(failed_names, failed_values, color="red")
            ax.axvline(x=threshold, linestyle="--", color="black", linewidth=1)
            ax.set_xlabel("signal within parcel %", fontsize=9)
            ax.set_xlim(0, threshold + 0.1)
            ax.set_title(
                f"{atlas} — Failed Parcels ({len(failed_names)}/{len(parcel_values)})",
                fontsize=9,
            )
            fig.tight_layout()

            _, center, _ = st.columns([1, 2, 1])
            with center:
                st.pyplot(fig, width='content')
            plt.close(fig)
            
    if num_atlas_failed:
        status_color = "red"
        failed_list = ", ".join(num_atlas_failed)
        st.markdown(f"**:{status_color}[FAIL]** — `{failed_list}` | ")
    else:
        status_color = "green"
        st.markdown(f"**:{status_color}[PASS]**")    
            

    # Store computed results in session state so collect_qc_record can save them
    state_key = f"{sub_id}_{ses}_{task}_{run}_coverage_results"
    st.session_state[state_key] = qc_results
    st.divider()


def render_pagination(total_rows: int) -> None:
    """
    Renders page-size selector + prev/next navigation at the bottom of the page.
    Mutates st.session_state directly; callers should check for st.rerun() needs.
    """
    bottom_menu = st.columns((1, 2, 1))

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
        if new_batch_size != st.session_state.batch_size:
            st.session_state.batch_size = new_batch_size
            st.session_state.current_page = 1
            st.rerun()

    import math
    total_pages = max(1, math.ceil(total_rows / st.session_state.batch_size))

    with bottom_menu[1]:
        col1, col2, col3 = st.columns([1, 1, 1], gap="small")

        if col1.button("⬅️"):
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
            # Signal to the caller to save before advancing
            st.session_state["_save_and_advance"] = True
            st.rerun()

    with bottom_menu[0]:
        st.markdown(f"Page **{st.session_state.current_page}** of **{total_pages}**")

    return total_pages


def render_scroll_to_top() -> None:
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


# Helper: collect QCRecord from session state after rendering

def collect_qc_record(
    sub_id: str,
    ses: Optional[str],
    task: Optional[str],
    run: Optional[str],
    metric_ids: List[str],
    qc_configs: list,
    rater_name: str,
    pipeline: str,
) -> QCRecord:
    """
    Reads widget values from st.session_state and assembles a QCRecord.
    Call this AFTER all fragments for a bundle have been rendered.
    """
    bundle_metrics: List[MetricQC] = []

    for metric_id in metric_ids:
        key_parts = [sub_id]
        if ses:  key_parts.append(ses)
        if task: key_parts.append(task)
        if run:  key_parts.append(run)
        key_parts.append(metric_id)
        widget_key = "_".join(key_parts)

        qc_val = st.session_state.get(widget_key)
        bundle_metrics.append(MetricQC(name=metric_id, qc=qc_val))

    rerun_key = f"{sub_id}_{ses}_{task}_{run}_rerun"
    notes_key = f"{sub_id}_{ses}_{task}_{run}_notes"

    require_rerun = st.session_state.get(rerun_key)
    notes         = st.session_state.get(notes_key, "")
    final_qc      = "FAIL" if require_rerun == "YES" else ("PASS" if require_rerun == "NO" else None)

    bundle_metrics.append(MetricQC(name="QC_notes", notes=notes))

    # Append auto-computed FD metrics at threshold=0.3
    fd_key = f"{sub_id}_{ses}_{task}_{run}_fd_metrics"
    for name, val in st.session_state.get(fd_key, {}).items():
        if name == "fd_pass":
            bundle_metrics.append(MetricQC(name="fd_pass", qc=val))
        else:
            bundle_metrics.append(MetricQC(name=name, value=val))

    return QCRecord(
        subject_id=sub_id,
        session_id=ses,
        task_id=task,
        run_id=run,
        pipeline=pipeline,
        complete_timestamp=None,
        rater=rater_name,
        require_rerun=require_rerun,
        final_qc=final_qc,
        metrics=bundle_metrics,
    )