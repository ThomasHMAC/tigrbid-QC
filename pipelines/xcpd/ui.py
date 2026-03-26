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

import streamlit as st

from models import MetricQC, QCRecord
from utils import QCEntry, QCKey


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
                st.markdown(_svg_as_img_tag(str(svg_path)), unsafe_allow_html=True)

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
    pipeline: str = "xcpd-0.7.3",
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