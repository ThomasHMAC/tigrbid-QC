#!/usr/bin/env python3
import sys
from pathlib import Path

UI_DIR = Path(__file__).resolve().parents[2] / "ui"
if str(UI_DIR) not in sys.path:
    sys.path.insert(0, str(UI_DIR))

import argparse
import math
import re
from datetime import date
from typing import Dict, Optional, Tuple, List, Any

import pandas as pd
import streamlit as st
from PIL import Image

from models import MetricQC, QCStatusRow

# -----------------------------
# Args / IO helpers
# -----------------------------
def parse_args(args=None):
    parser = argparse.ArgumentParser(description="AMICO NODDI QC")
    parser.add_argument("--noddireg_dir", required=True)
    parser.add_argument("--participant_labels", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args(args)


def load_participants_df(tsv: str) -> pd.DataFrame:
    df = pd.read_csv(tsv, sep="\t")
    if "participant_id" not in df.columns:
        raise ValueError("participants.tsv must contain participant_id")
    return df


def detect_sessions(subj_dir: Path, subject: str) -> List[str]:
    sessions = set()
    for f in subj_dir.glob(f"{subject}_*noddi*.png"):
        m = re.search(r"(ses-[a-zA-Z0-9]+)", f.name)
        sessions.add(m.group(1) if m else "no-session")
    return sorted(sessions)


def do_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        st.warning("No rerun function available in this Streamlit version.")


def init_session_state():
    if "current_page" not in st.session_state:
        st.session_state.current_page = 1


def get_current_batch(df: pd.DataFrame, current_page: int):
    batch_size = 1
    total_rows = len(df)
    start_idx = (current_page - 1) * batch_size
    end_idx = min(start_idx + batch_size, total_rows)
    return total_rows, df.iloc[start_idx:end_idx]


# -----------------------------
# UI helpers
# -----------------------------
QC_OPTIONS = ("pass", "fail", "uncertain")


def seed_from_saved(key: str, saved_val):
    if saved_val is None:
        return
    if key not in st.session_state:
        st.session_state[key] = saved_val
        return
    cur = st.session_state.get(key)
    if cur is None:
        st.session_state[key] = saved_val
        return
    if isinstance(cur, str) and cur.strip() == "":
        st.session_state[key] = saved_val
        return
    if key.endswith("_qc") and cur not in QC_OPTIONS:
        st.session_state[key] = saved_val
        return


def qc_radio(label: str, key: str, stored_val: Optional[str]):
    if stored_val not in QC_OPTIONS:
        stored_val = None
    seed_from_saved(key, stored_val)
    idx = (
        QC_OPTIONS.index(st.session_state[key])
        if st.session_state.get(key) in QC_OPTIONS
        else None
    )
    return st.radio(label, QC_OPTIONS, key=key, horizontal=True, index=idx)


def notes_box(label: str, key: str, stored_val: Optional[str]):
    # notes should be blank if nothing was saved
    stored_val = "" if stored_val is None else str(stored_val)
    seed_from_saved(key, stored_val)
    return st.text_area(label, key=key)


def compute_overall(metrics: List[MetricQC]) -> Optional[str]:
    qcs = [m.qc for m in metrics if m.qc in QC_OPTIONS]
    if not qcs:
        return None
    if "fail" in qcs:
        return "fail"
    if "uncertain" in qcs:
        return "uncertain"
    return "pass"


# -----------------------------
# TSV persistence (qc_status.tsv pattern)
# -----------------------------
Key = Tuple[str, Optional[str], Optional[str], Optional[int], str]  # pid, session, acq, run, qc_task

OUTPUT_COLUMNS = [
    "participant_id",
    "session",
    "acq",
    "run",
    "qc_task",
    "rater_id",
    "score",
    "notes",
    "timestamp",
]


def _norm_str(x) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, float) and pd.isna(x):
        return None
    s = str(x).strip()
    return None if s == "" or s.lower() == "none" else s


def _norm_int(x) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, float) and pd.isna(x):
        return None
    if isinstance(x, int):
        return x
    s = str(x).strip()
    if s == "" or s.lower() == "none":
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def _norm_date_str(x) -> Optional[str]:
    """
    We store timestamp as YYYY-MM-DD string in TSV.
    """
    s = _norm_str(x)
    if not s:
        return None
    return s.split("T")[0].split(" ")[0]


def read_existing_tsv(out_file: Path) -> pd.DataFrame:
    if not out_file.exists():
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    try:
        df = pd.read_csv(out_file, sep="\t", dtype=str)
    except Exception:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    for c in OUTPUT_COLUMNS:
        if c not in df.columns:
            df[c] = ""

    return df[OUTPUT_COLUMNS].copy()


def build_lookup(out_file: Path) -> Dict[Key, Dict[str, Any]]:
    """
    Lookup used ONLY to seed:
      - score per qc_task
      - overall notes (from overall_task row)
      - rater_id per qc_task
    """
    df = read_existing_tsv(out_file)
    if df.empty:
        return {}

    lookup: Dict[Key, Dict[str, Any]] = {}

    for _, r in df.iterrows():
        pid = _norm_str(r.get("participant_id"))
        ses = _norm_str(r.get("session"))
        acq = _norm_str(r.get("acq"))
        run = _norm_int(r.get("run"))
        qc_task = _norm_str(r.get("qc_task"))
        if not pid or not qc_task:
            continue

        score = _norm_str(r.get("score"))
        score = score if score in QC_OPTIONS else None

        notes_raw = r.get("notes")
        notes = "" if notes_raw is None or (isinstance(notes_raw, float) and pd.isna(notes_raw)) else str(notes_raw)

        rater = _norm_str(r.get("rater_id")) or ""

        lookup[(pid, ses, acq, run, qc_task)] = {
            "score": score,
            "notes": notes,
            "rater_id": rater,
        }

    return lookup


def lookup_val(
    lookup: Dict[Key, Dict[str, Any]],
    pid: str,
    ses: Optional[str],
    acq: Optional[str],
    run: Optional[int],
    qc_task: str,
):
    d = lookup.get((pid, ses, acq, run, qc_task), {})
    score = d.get("score")
    score = score if score in QC_OPTIONS else None
    notes = d.get("notes", "") 
    rater = d.get("rater_id", "")
    return score, notes, rater


def qcstatusrows_to_df(rows: List[QCStatusRow]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    raw = []
    for r in rows:
        d = r.model_dump()

        ts = d.get("timestamp")
        if isinstance(ts, date):
            d["timestamp"] = ts.isoformat()  # YYYY-MM-DD
        elif ts is None:
            d["timestamp"] = ""
        else:
            d["timestamp"] = _norm_date_str(ts) or ""

        if d.get("notes") is None:
            d["notes"] = ""

        raw.append(d)

    df = pd.DataFrame(raw)

    for c in OUTPUT_COLUMNS:
        if c not in df.columns:
            df[c] = ""

    for c in ["participant_id", "session", "acq", "qc_task", "rater_id", "score", "notes", "timestamp"]:
        df[c] = df[c].fillna("").astype(str)

    if "run" in df.columns:
        df["run"] = df["run"].replace({"None": "", "nan": ""}).fillna("").astype(str)

    return df[OUTPUT_COLUMNS].copy()


def save_rows_overwrite_tsv(out_file: Path, new_rows: List[QCStatusRow]) -> Path:
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    old = read_existing_tsv(out_file)
    new_df = qcstatusrows_to_df(new_rows)

    if new_df.empty and old.empty:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(out_file, sep="\t", index=False)
        return out_file

    if new_df.empty:
        old.to_csv(out_file, sep="\t", index=False)
        return out_file

    subset = ["participant_id", "session", "acq", "run", "qc_task"]

    def make_key(df: pd.DataFrame) -> pd.Series:
        tmp = df.copy()
        for c in subset:
            if c not in tmp.columns:
                tmp[c] = ""
            tmp[c] = tmp[c].fillna("").astype(str)
        return tmp[subset].agg("||".join, axis=1)

    new_keys = set(make_key(new_df))

    if not old.empty:
        old_keys = make_key(old)
        old = old[~old_keys.isin(new_keys)].copy()

    out = pd.concat([old, new_df], ignore_index=True)

    for c in ["participant_id", "session", "acq", "run", "qc_task"]:
        if c not in out.columns:
            out[c] = ""
    out = out.sort_values(by=["participant_id", "session", "acq", "run", "qc_task"]).reset_index(drop=True)

    out.to_csv(out_file, sep="\t", index=False)
    return out_file


# -----------------------------
# Main app
# -----------------------------
def main():
    args = parse_args()

    noddireg_dir = Path(args.noddireg_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    participants_df = load_participants_df(args.participant_labels)

    st.set_page_config(layout="centered")
    st.markdown('<div id="top_of_page"></div>', unsafe_allow_html=True)
    st.title("AMICO NODDI QC")

    init_session_state()

    out_file = output_dir / "noddireg_qc_status.tsv"
    lookup = build_lookup(out_file)

    rater_name = st.text_input("Rater name (optional):", value="")
    typed_rater_id = (rater_name or "").strip()

    total_rows, current_batch = get_current_batch(participants_df, st.session_state.current_page)
    total_pages = max(1, math.ceil(total_rows / 1))

    qc_rows: List[QCStatusRow] = []

    for _, prow in current_batch.iterrows():
        subject = str(prow["participant_id"])
        st.sidebar.markdown(f"### {subject}")

        subj_dir = noddireg_dir / subject
        if not subj_dir.exists():
            st.error(f"Missing directory: {subj_dir}")
            continue

        sessions = detect_sessions(subj_dir, subject)
        if not sessions:
            st.warning(f"No NODDI images found for {subject}")
            continue

        for ses in sessions:
            st.divider()
            st.header(f"{subject} — {ses}")

            ses_prefix = f"{subject}_{ses}" if ses != "no-session" else subject
            session_id = None if ses == "no-session" else ses

            acq_id: Optional[str] = None
            run_id: Optional[int] = None

            metrics_for_overall: List[MetricQC] = []

            # ---- Density ----
            st.subheader("Tissue Density (CSF / GM / WM)")
            density_pngs = list(subj_dir.glob(f"{ses_prefix}*_desc-dsegtissue_model-noddi_density.png"))
            if density_pngs:
                for png in density_pngs:
                    st.image(Image.open(png), use_container_width=True)
                    st.caption(str(png))
            else:
                st.warning("Density plot not found")

            density_task = "noddireg_density"
            stored_score, _, stored_rater = lookup_val(lookup, subject, session_id, acq_id, run_id, density_task)
            effective_rater_id = typed_rater_id if typed_rater_id != "" else stored_rater

            density_qc_key = f"{subject}_{ses}_{density_task}_qc"
            density_score = qc_radio("Density QC", density_qc_key, stored_score)

            metrics_for_overall.append(MetricQC(name="density", qc=density_score))

            qc_rows.append(
                QCStatusRow(
                    participant_id=subject,
                    session=session_id,
                    acq=acq_id,
                    run=run_id,
                    qc_task=density_task,
                    rater_id=effective_rater_id,
                    score=density_score,
                    notes=None,
                    timestamp=date.today(),
                )
            )

            st.divider()

            # ---- Parcel-wise metrics ----
            st.subheader("Parcel-wise NODDI Metrics (Schaefer 4S1056)")
            metric_specs = [
                ("od_mean", "noddireg_od", "od", "OD"),
                ("icvf_mean", "noddireg_icvf", "icvf", "ICVF"),
                ("isovf_mean", "noddireg_isovf", "isovf", "ISOVF"),
            ]

            for metric_fname, qc_task, metric_name, label in metric_specs:
                st.markdown(f"**{label}**")

                pngs = sorted(subj_dir.glob(f"{ses_prefix}*_{metric_fname}_qc.png"))
                if pngs:
                    for png in pngs:
                        st.image(Image.open(png), use_container_width=True)
                        st.caption(str(png))
                else:
                    st.warning(f"{label} QA plot not found")

                stored_score_m, _, stored_rater_m = lookup_val(lookup, subject, session_id, acq_id, run_id, qc_task)
                effective_rater_m = typed_rater_id if typed_rater_id != "" else stored_rater_m

                qc_key = f"{subject}_{ses}_{qc_task}_qc"
                score_val = qc_radio(f"{label} QC", qc_key, stored_score_m)

                metrics_for_overall.append(MetricQC(name=metric_name, qc=score_val))

                qc_rows.append(
                    QCStatusRow(
                        participant_id=subject,
                        session=session_id,
                        acq=acq_id,
                        run=run_id,
                        qc_task=qc_task,
                        rater_id=effective_rater_m,
                        score=score_val,
                        notes=None,
                        timestamp=date.today(),
                    )
                )

                st.divider()

            # ---- Overall (one notes field only) ----
            overall_task = "noddireg_overall"
            overall_stored_score, overall_stored_notes, overall_stored_rater = lookup_val(
                lookup, subject, session_id, acq_id, run_id, overall_task
            )
            effective_rater_overall = typed_rater_id if typed_rater_id != "" else overall_stored_rater

            overall_auto = compute_overall(metrics_for_overall)
            overall_final = overall_auto if overall_auto is not None else overall_stored_score

            overall_notes_key = f"{subject}_{ses}_{overall_task}_notes"
            overall_notes = notes_box("Overall notes", overall_notes_key, overall_stored_notes)
            overall_notes = (overall_notes or "").strip()
            overall_notes_out = overall_notes if overall_notes != "" else None  # blank by default

            qc_rows.append(
                QCStatusRow(
                    participant_id=subject,
                    session=session_id,
                    acq=acq_id,
                    run=run_id,
                    qc_task=overall_task,
                    rater_id=effective_rater_overall,
                    score=overall_final,
                    notes=overall_notes_out,
                    timestamp=date.today(),    
                )
            )

    def save_now():
        save_rows_overwrite_tsv(out_file, qc_rows)
        st.success(f"Saved: {out_file}")

    if st.button("💾 Save now"):
        save_now()

    bottom = st.columns((1, 2, 1))

    with bottom[0]:
        st.markdown(f"Page **{st.session_state.current_page}** of **{total_pages}**")

    with bottom[1]:
        c1, c2, c3 = st.columns([1, 1, 1], gap="small")

        if c1.button("⬅️"):
            save_now()
            if st.session_state.current_page > 1:
                st.session_state.current_page -= 1
            do_rerun()

        new_page = c2.number_input(
            "Page",
            min_value=1,
            max_value=total_pages,
            value=st.session_state.current_page,
            step=1,
            key="page_jump",
        )
        if new_page != st.session_state.current_page:
            save_now()
            st.session_state.current_page = int(new_page)
            do_rerun()

        if c3.button("➡️"):
            save_now()
            if st.session_state.current_page < total_pages:
                st.session_state.current_page += 1
            do_rerun()

    with bottom[2]:
        st.markdown("")

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


if __name__ == "__main__":
    main()
