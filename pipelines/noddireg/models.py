from datetime import date
from typing import Annotated, List, Optional, Dict, Literal
from pathlib import Path

from pydantic import BaseModel, Field, RootModel


# Future plans:
# To be used if we want to provide configurable QC scoring options
class MetricQC(BaseModel):
    name: Annotated[
        str, Field(description="Name of the metric, e.g., Euler, segmentation")
    ]
    value: Annotated[
        Optional[float], Field(description="Numeric value if applicable")
    ] = None
    qc: Annotated[
        Optional[str], Field(description="QC decision: PASS, FAIL, UNCERTAIN")
    ] = None
    notes: Annotated[Optional[str], Field(description="Additional comment")] = None


class QCRecord(BaseModel):
    qc_task: Annotated[str, Field(description="QC task identifier, e.g., sdc-wf")]
    participant_id: Annotated[str, Field(description="BIDS subject ID")]
    session_id: Annotated[str, Field(description="Session ID, e.g., ses-01")]
    task_id: Optional[str] = None
    run_id: Optional[str] = None
    pipeline: Annotated[
        str, Field(description="Pipeline name and version, e.g., freesurfer")
    ]
    timestamp: Annotated[
        Optional[str], Field(description="Completion date")
    ] = None
    rater_id: Annotated[str, Field(description="Name of the rater")]
    rater_experience: Annotated[
        Optional[str], Field(description="Rater experience level")
    ] = None
    rater_fatigue: Annotated[
        Optional[str], Field(description="Rater fatigue level")
    ] = None
    final_qc: Optional[str] = None
    notes: Annotated[Optional[str], Field(description="Additional comment")] = None


class QCTask(BaseModel):
    """Represents one QC entry in <pipeline>_qc.json (i.e. single QC task)."""

    base_mri_image_path: Annotated[
        Optional[Path], Field(description="Path to base MRI image")
    ] = None

    overlay_mri_image_path: Annotated[
        Optional[Path], Field(description="Path to overlay MRI image (mask etc.)")
    ] = None

    # Updated to list to match the repo plan (can show multiple montages)
    svg_montage_path: Annotated[
        Optional[List[Path]], Field(description="Path(s) to SVG montage(s) for visual QC")
    ] = None

    # Updated to list to match the repo plan (can load multiple IQM files)
    iqm_path: Annotated[
        Optional[List[Path]], Field(description="Path(s) to IQM TSV/JSON or other QC files")
    ] = None


class QCConfig(RootModel[Dict[str, QCTask]]):
    """Top-level model for `qc.json`.

    The JSON is expected to be a mapping from QC-task keys (strings) to
    `QCTask` objects. Example:

    {
        "anat_wf_qc": {
            "base_mri_image_path": "...",
            "overlay_mri_image_path": "...",
            "svg_montage_path": ["...svg", "...svg"],
            "iqm_path": ["...tsv"]
        }
    }
    """
    pass


# -----------------------------
# qc_status.tsv model
# -----------------------------

QCDecision = Literal["pass", "fail", "uncertain"]


class QCStatusRow(BaseModel):
    participant_id: Annotated[str, Field(description="BIDS subject ID, e.g., sub-ED01")]
    session: Optional[str] = None
    acq: Optional[str] = None
    run: Optional[int] = None
    qc_task: Annotated[str, Field(description="QC task identifier, e.g., anat_wf_qc")]
    rater_id: Annotated[str, Field(description="Rater identifier")]
    score: Optional[QCDecision] = None
    notes: Optional[str] = None
    timestamp: Optional[date] = None
