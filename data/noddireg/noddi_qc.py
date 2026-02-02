#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd
import streamlit as st
from PIL import Image
import re


# -----------------------------
# Argument parsing
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="AMICO NODDI QC")
    parser.add_argument("--noddireg_dir", required=True)
    parser.add_argument("--participant_labels", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


# -----------------------------
# Participants
# -----------------------------
def load_participants(tsv):
    df = pd.read_csv(tsv, sep="\t")
    if "participant_id" not in df.columns:
        raise ValueError("participants.tsv must contain participant_id")
    return df["participant_id"].tolist()


# -----------------------------
# Detect sessions from filenames
# -----------------------------
def detect_sessions(subj_dir, subject):
    sessions = set()
    for f in subj_dir.glob(f"{subject}_*noddi*.png"):
        m = re.search(r"(ses-[a-zA-Z0-9]+)", f.name)
        sessions.add(m.group(1) if m else "no-session")
    return sorted(sessions)


# -----------------------------
# Streamlit app
# -----------------------------
def main():
    args = parse_args()

    noddireg_dir = Path(args.noddireg_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    participants = load_participants(args.participant_labels)

    st.set_page_config(layout="centered")
    st.title("AMICO NODDI QC")

    # -----------------------------
    # Session state: subject index
    # -----------------------------
    if "subj_idx" not in st.session_state:
        st.session_state.subj_idx = 0

    subject = participants[st.session_state.subj_idx]

    st.sidebar.markdown(f"**Subject {st.session_state.subj_idx + 1} / {len(participants)}**")
    st.sidebar.markdown(f"### {subject}")

    subj_dir = noddireg_dir / subject
    if not subj_dir.exists():
        st.error(f"Missing directory: {subj_dir}")
        return

    sessions = detect_sessions(subj_dir, subject)

    if not sessions:
        st.warning("No NODDI images found")
        return

    out_rows = []

    for ses in sessions:
        st.divider()
        st.header(f"{subject} — {ses}")

        ses_prefix = f"{subject}_{ses}" if ses != "no-session" else subject

        # -----------------------------
        # Density plot
        # -----------------------------
        st.subheader("Tissue Density (CSF / GM / WM)")

        density_pngs = list(
            subj_dir.glob(f"{ses_prefix}*_desc-dsegtissue_model-noddi_density.png")
        )

        if density_pngs:
            for png in density_pngs:
                st.image(Image.open(png), use_container_width=True)
                st.caption(str(png))
        else:
            st.warning("Density plot not found")

        density_qc = st.radio(
            "Density QC",
            ["PASS", "FAIL", "UNCERTAIN"],
            horizontal=True,
            key=f"{subject}_{ses}_density_qc",
        )

        density_comment = st.text_area(
            "Density comments",
            key=f"{subject}_{ses}_density_comment",
        )

        st.divider()

        # -----------------------------
        # Parcel-wise QA
        # -----------------------------
        st.subheader("Parcel-wise NODDI Metrics (Schaefer 4S1056)")

        metrics = ["od_mean", "icvf_mean", "isovf_mean"]
        metric_qc = {}
        metric_comment = {}

        for metric in metrics:
            st.markdown(f"**{metric.upper()}**")

            pngs = sorted(
                subj_dir.glob(
                    f"{ses_prefix}*_{metric}_qc.png"
                )
            )

            if pngs:
                for png in pngs:
                    st.image(Image.open(png), use_container_width=True)
                    st.caption(str(png))
            else:
                st.warning(f"{metric.upper()} QA plot not found")

            metric_qc[metric] = st.radio(
                f"{metric.upper()} QC",
                ["PASS", "FAIL", "UNCERTAIN"],
                horizontal=True,
                key=f"{subject}_{ses}_{metric}_qc",
            )

            metric_comment[metric] = st.text_area(
                f"{metric.upper()} comments",
                key=f"{subject}_{ses}_{metric}_comment",
            )

            st.divider()

        out_rows.append(
            {
                "participant_id": subject,
                "session": ses,
                "density_qc": density_qc,
                "density_comment": density_comment,
                "od_qc": metric_qc["od_mean"],
                "od_comment": metric_comment["od_mean"],
                "icvf_qc": metric_qc["icvf_mean"],
                "icvf_comment": metric_comment["icvf_mean"],
                "isovf_qc": metric_qc["isovf_mean"],
                "isovf_comment": metric_comment["isovf_mean"]
            }
        )

    # -----------------------------
    # Save + Next
    # -----------------------------
    if st.button("💾 Save & Next ▶️"):
        out_tsv = output_dir / "noddi_qc.tsv"

        new_df = pd.DataFrame(out_rows)

        if out_tsv.exists():
            old = pd.read_csv(out_tsv, sep="\t")
            old = old[old.participant_id != subject]
            out = pd.concat([old, new_df], ignore_index=True)
        else:
            out = new_df

        out.to_csv(out_tsv, sep="\t", index=False)

        if st.session_state.subj_idx < len(participants) - 1:
            st.session_state.subj_idx += 1
            st.experimental_rerun()
        else:
            st.success("🎉 All subjects QCed!")


if __name__ == "__main__":
    main()
