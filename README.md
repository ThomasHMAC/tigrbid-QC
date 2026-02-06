# Instruction to run streamlit app for Kimel 
1. Create python environment and cloning the tigrbid-QC repo
```bash
cd /projects/ttan
module load python/3.10.7
python -m venv tigrbid_QC_env
source tigrbid_QC_env/bin/activate
pip install streamlit==1.49.1 pandas==2.3.2 pydantic==2.11.7 beautifulsoup4

git clone https://github.com/ThomasHMAC/tigrbid-QC.git
```

2. Run freesurfer QC dashboard

```bash
cd /projects/ttan/tigrbid-QC

streamlit run ./pipelines/freesurfer/fs_main.py -- --fs_metric /projects/galinejad/SCanD_CAMH_SPINS/share/freesurfer_group/euler.tsv --fmri_dir /projects/galinejad/SCanD_CAMH_SPINS/share/fmriprep/23.2.3/ --participant_labels /projects/ttan/tigrbid-QC/SPINS_participants.tsv --output_dir /projects/ttan/tigrbid-QC/outputs/SPINS_QC/
```

3. Run fmriprep QC dashboard

```bash
cd /projects/ttan/tigrbid-QC
streamlit run ./pipelines/fmriprep/fmriprep_main.py -- --fmri_dir /projects/galinejad/SCanD_CAMH_RTMSWM/share/fmriprep/23.2.3 --participant_labels /projects/ttan/RTMSWM/participants.tsv --output_dir /projects/ttan/tigrbid-QC/outputs/RTMSWM_QC/
```

4. Run Noddireg QC dashboard
```bash
cd /projects/galinejad/tigrbid-QC
streamlit run ./pipelines/noddireg/noddi_qc.py --   --noddireg_dir /projects/galinejad/SCanD_CAMH_RTMSWM/share/noddireg    --participant_labels /projects/galinejad/SCanD_CAMH_RTMSWM/share/participants.tsv   --output_dir /projects/galinejad/tigrbid-QC/outputs/RTMSWM_QC/
```

5. Run streamlit remotely through ssh tunnel
```bash
ssh -X -L 8501:localhost:8501 ttan@darwin.camhres.ca
cd /projects/ttan/tigrbid-QC
streamlit run ./pipelines/freesurfer/fs_main.py --server.port=8501 -- \
  --fs_metric /projects/galinejad/SCanD_CAMH_SPINS/share/freesurfer_group/euler.tsv \
  --fmri_dir /projects/galinejad/SCanD_CAMH_SPINS/share/fmriprep/23.2.3/ \
  --participant_labels /projects/ttan/tigrbid-QC/SPINS_participants.tsv \
  --output_dir /projects/ttan/tigrbid-QC/outputs/SPINS_QC/
```

Once you run this you can open a browser on your computer and paste in the http://localhost:8501 URL
