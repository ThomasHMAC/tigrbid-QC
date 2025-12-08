# %%
import math
import os
import re
from argparse import ArgumentParser
from datetime import datetime
from glob import glob
from pathlib import Path

import pandas as pd
import streamlit as st
from models import MetricQC, QCRecord
# from session_persistance import load_session_state, save_session_state
# from streamlit_scroll_to_top import scroll_to_here

# Debug: show keys on every rerun
# st.write("Session state before initialized:", st.session_state)

st.markdown('<div id="top_of_page"></div>', unsafe_allow_html=True)

def parse_args(args=None):
    parser = ArgumentParser("Freesurfer QC")

    parser.add_argument(
        "--fs_metric",
        dest="freesurfer_metric",
        help="Group Euler CSV file",
        required=True,
    )

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
fs_metric = args.freesurfer_metric
participant_labels = args.participant_labels
fmri_dir = args.fmri_dir
out_dir = args.out_dir

# fs_metric = "/projects/ttan/RTMSWM/for_jackie/freesurfer_group/euler.tsv"
# fs_metric = "/projects/galinejad/SCanD_CAMH_SPINS/share/freesurfer_group/euler.tsv"
# fs_metric = "/projects/ttan/CDIA/data/local/derivatives/euler.tsv"
# fs_metric = "/projects/ttan/TAY/derivatives/freesurfer/7.4.1/00_group2_stats_tables/euler.tsv"
# participant_labels = "/projects/ttan/tigrbid-QC/TAY_participants.tsv"
# out_dir = "/projects/ttan/tigrbid-QC/outputs/TAY_QC/"

participants_df = pd.read_csv(participant_labels, delimiter="\t")

def get_fs_metrics(euler_path):
    fs_metrics_df = pd.read_csv(
        euler_path,
        sep="\t",
    )
    # Detect whether the dataset has ANY sessions
    is_longitudinal = fs_metrics_df["subject"].str.contains("ses-").any()

    if is_longitudinal:
        fs_metrics_df = fs_metrics_df[
            fs_metrics_df["subject"].str.contains("ses-")
        ].copy()
        fs_metrics_df[["participant_id", "session_id"]] = \
            fs_metrics_df["subject"].str.split("_", expand=True)
    else:
        fs_metrics_df = fs_metrics_df.copy()
        fs_metrics_df["participant_id"] = fs_metrics_df["subject"]
        fs_metrics_df["session_id"] = "cross-sectional"
    return fs_metrics_df

fs_metrics_df = get_fs_metrics(fs_metric)

def scroll():
    st.session_state.scroll_to_top = True


def scrollheader():
    st.session_state.scroll_to_header = True


st.title("Freesurfer QC")
rater_name = st.text_input("Rater name:")
# Show the value dynamically
st.write("You entered:", rater_name)


def init_session_state():
    defaults = {
        "current_page": 1,
        "batch_size": 10,
        "scroll_to_top": False,
        "scroll_to_header": False,
        "current_batch_qc": {},
    }
    # Initialize defaults if not already set
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Handle scrolling logic
    if st.session_state.scroll_to_top:
        scroll_to_here(0, key="top")
        st.session_state.scroll_to_top = False


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
            "run": rec.run_id,
            "pipeline": rec.pipeline,
            "recon_timestamp": rec.complete_timestamp,
        }

        for m in rec.metrics:
            metric_name = m.name.lower().replace("-", "_")
            if m.value is not None:
                row[f"{metric_name}_value"] = m.value
            if m.qc is not None:
                row[f"{metric_name}_qc"] = m.qc

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
            subset=["subject", "session", "run", "pipeline"], keep="last"
        )
    df = df.sort_values(by=["subject"]).reset_index(drop=True)
    df.to_csv(out_file, index=False)

    return out_file


def get_metrics_from_csv(qc_results: Path, metrics_to_load=None):
    if metrics_to_load is None:
        metrics_to_load = ["surface_segmentation_qc", "require_rerun"]
    if qc_results.exists():
        df_existing = pd.read_csv(qc_results)
        # Keep only columns that actually exist
        columns_available = [
            col for col in metrics_to_load if col in df_existing.columns
        ]

        if "subject" not in df_existing.columns or not columns_available:
            return {}
        if "session" not in df_existing.columns:
            df_existing["session"] = "ses-01"

        # Drop exact duplicate rows (same subject + session)
        df_existing = df_existing.drop_duplicates(
            subset=["subject", "session"], keep="last"
        )
        # Set a multi-index
        df_existing.set_index(["subject", "session"], inplace=True)

        data_dict = df_existing[columns_available].to_dict(orient="index")
    else:
        data_dict = {}

    def get_val(sub_id, ses_id, metric):
        key_sub = sub_id if sub_id.startswith("sub-") else f"sub-{sub_id}"
        if ses_id == "cross-sectional":
            key_ses = "cross-sectional"
        else:
            key_ses = ses_id if ses_id.startswith("ses-") else f"ses-{ses_id}"
        return data_dict.get((key_sub, key_ses), {}).get(metric, None)

    return metrics_to_load, get_val

# def clear_old_qc_state():
#     for key in list(st.session_state.keys()):
#         if any(x in key for x in ["_seg", "_euler", "_notes", "_rerun"]):
#             del st.session_state[key]


# total_rows, current_batch = get_current_batch(
#     fs_metrics_df, st.session_state.current_page, st.session_state.batch_size
# )
total_rows, current_batch = get_current_batch(
    participants_df, st.session_state.current_page, st.session_state.batch_size
)
merged_batch = current_batch.merge(
    fs_metrics_df,
    on="participant_id",
    how="left"
)

# Save to CSV
# now = datetime.now()
# timestamp = now.strftime("%Y%m%d")  # e.g., 20250917
# out_file = Path(out_dir) / f"qc_results_{timestamp}.csv"
out_file = Path(out_dir) / f"freesurfer_QC_status.csv"

data_dict, get_val = get_metrics_from_csv(out_file)
# Collect all the current batch subject metrics
qc_records = []
displayed_subjects = set()
for _, row in merged_batch.iterrows():
    pid = row["participant_id"]
    sub_id = pid.split("-")[1]
    ses_id = row["session_id"]
    if pd.isna(ses_id):
        st.warning(f"No FreeSurfer metrics found for {pid}")
        continue

    if ses_id.startswith("ses-"):
        ses_num = ses_id.split("-")[1]
    else:
        ses_num = ses_id
    # ses_num = ses_id.split("-")[1]
    run_id = None

    # subj = row["subject"]
    # parts = subj.split("_")
    # sub_id = parts[0].split("-")[1]
    # ses_id = parts[1] if len(parts) > 1 else "ses-01"
    # ses_num = ses_id.split("-")[1] if "-" in ses_id else ses_id
    # run_id = None

    # metrics per subject
    subject_metrics = []

    st.header(f"Subject {sub_id} Session {ses_num}")
    # show SVG once per subject (not per session)
    if sub_id not in displayed_subjects:
        svg_matches = glob(
            f"{fmri_dir}/sub-{sub_id}/figures/sub-{sub_id}_*desc-reconall_T1w.svg"
        )
        svg_path = svg_matches[0] if svg_matches else None
        st.set_page_config(layout="wide")

        with st.container():
            if svg_path is not None and os.path.exists(svg_path):
                st.image(svg_path, use_container_width=True)
                st.write(f"**{svg_path}**")
            else:
                st.warning(f"Image not found: {svg_path}")
            st.markdown(f"<h4> Surface Segmentation QC", unsafe_allow_html=True)
        displayed_subjects.add(sub_id)

        # --- Surface segmentation QC radio ---
        metric = "surface_segmentation_qc"
        sub_val = get_val(f"sub-{sub_id}", ses_id, metric)
        options = ("PASS", "FAIL", "UNCERTAIN")
        seg_qc = st.radio(
            "",
            options,
            key=f"{sub_id}_seg",
            label_visibility="collapsed",
            index=options.index(sub_val) if sub_val in options else None,
        )

    else:
        # Reuse previously chosen QC value for this subject
        seg_qc = st.session_state.get(f"{sub_id}_seg")

    # Add the segmentation QC result (same for all sessions)
    metric = MetricQC(name="surface_segmentation", qc=seg_qc)
    subject_metrics.append(metric)

    # Get recon-all timestamp
    log_file = Path(
        f"{fmri_dir}/sourcedata/freesurfer/sub-{sub_id}/scripts/recon-all-status.log"
    )
    complete_time = None
    formatted_date = None
    if log_file.is_file():
        lines = log_file.read_text().splitlines()
        last_line = lines[-1]
        if "finished without error at" in last_line:
            finished_str = last_line.split(" at ")[1]
            finished_str_no_tz = " ".join(
                finished_str.split()[1:-2] + [finished_str.split()[-1]]
            )
            format_data = "%b %d %H:%M:%S %Y"
            complete_time = datetime.strptime(finished_str_no_tz, format_data)
            formatted_date = complete_time.strftime("%m-%d-%Y")
    else:
        st.warning(f"Log file not found for subject {sub_id}.")

    # Euler metrics
    euler_vals = {"Left": row.get("lh_euler"), "Right": row.get("rh_euler")}
    st.markdown(f"<h4>Euler QC (Session {ses_num})</h4>", unsafe_allow_html=True)
    for hemi, val in euler_vals.items():
        euler_key = f"{sub_id}_{ses_id}_euler_{hemi}"
        options = ("PASS", "FAIL", "UNCERTAIN")

        if val < -150 or val > 2:
            default_index = options.index("FAIL")
        elif -150 <= val <= 2:
            default_index = options.index("PASS")
        else:
            default_index = None
        st.markdown(f"<h4>{hemi} Euler value: {val}</h4>", unsafe_allow_html=True)

        st.radio(
            "",
            options=options,
            key=euler_key,
            label_visibility="collapsed",
            index=default_index,
        )

        qc_choice = st.session_state.get(euler_key)
        metric = MetricQC(name=f"Euler_{hemi}", value=val, qc=qc_choice)

        # Avoid duplicating entries on rerun
        subject_metrics.append(metric)

    # --- Require rerun QC radio ---
    metric = "require_rerun"
    sub_val = get_val(f"sub-{sub_id}", ses_id, metric)
    options = ("YES", "NO")
    require_rerun = st.radio(
        f"Require rerun?",
        options,
        key=f"{sub_id}_{ses_id}_rerun",
        index=options.index(sub_val) if sub_val in options else None,
    )
    if require_rerun is None:
        final_qc = None
    else:
        final_qc = "FAIL" if require_rerun == "YES" else "PASS"
    # Notes
    notes = st.text_input(f"***NOTES***", key=f"{sub_id}_{ses_id}_notes")
    subject_metrics.append(MetricQC(name="QC_notes", notes=notes))
    # Create QCRecord
    record = QCRecord(
        subject_id=sub_id,
        session_id=ses_id,
        run_id=run_id,
        pipeline="freesurfer-7.4.1",
        complete_timestamp=formatted_date,
        rater=rater_name,
        require_rerun=require_rerun,
        final_qc=final_qc,
        metrics=subject_metrics,
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

st.write("batch size is", st.session_state.batch_size)
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
            # clear_old_qc_state()
            st.session_state.current_page += 1
            st.rerun()  # Force rerun to update immediately

with bottom_menu[0]:
    st.markdown(f"Page **{st.session_state.current_page}** of **{total_pages}**")

# st.button("Scroll to Top", on_click=scroll)
# st.write("The current session state is:", qc_records)

# st.write("The current session state is:", sorted(st.session_state.items()))
st.write("The current session state is:", len(st.session_state))
# st.write("The current QC records is:", qc_records)
# st.write("The current total number of QC records is:", len(qc_records))
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