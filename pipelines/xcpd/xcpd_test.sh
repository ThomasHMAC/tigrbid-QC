#!/bin/bash
# in_dir="/archive/data/TAY/pipelines/in_progress/jwong/xcp"
# participant_labels="/projects/ttan/tigrbid-QC/TAY_participants.tsv"
# out_dir="/projects/ttan/tigrbid-QC/outputs/TAY_20260402/"
# qc_config="/projects/ttan/tigrbid-QC/pipelines/xcpd/xcpd-0.14.1_qc.json"

in_dir="/projects/ttan/ASCEND/data/derivatives/xcp_d/0.7.3"
participant_labels="/projects/ttan/ASCEND/data/bids/participants.tsv"
out_dir="/projects/ttan/tigrbid-QC/outputs/ASCEND_20260323/"
qc_config="/projects/ttan/tigrbid-QC/pipelines/xcpd/xcpd-0.7.3_qc.json"
streamlit run ./xcpd_qc.py -- --xcpd_dir ${in_dir} --participant_labels ${participant_labels} --output_dir ${out_dir} --qc_config ${qc_config}
