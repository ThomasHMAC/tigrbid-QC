from datetime import datetime
from typing import Annotated, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

# # Allowed QC decisions
# QCDecision = Literal["PASS", "FAIL", "UNCERTAIN"]


class MetricQC(BaseModel):
    name: Annotated[
        str, Field(description="Name of the metric, e.g., Euler, segmentation")
    ]
    value: Annotated[
        Optional[Union[float,str]], Field(description="Numeric or string value if applicable")
    ] = None
    qc: Annotated[
        Optional[str], Field(description="QC decision: PASS, FAIL, UNCERTAIN")
    ] = None
    notes: Annotated[Optional[str], Field(description="Additional comment")] = None


class QCRecord(BaseModel):
    subject_id: Annotated[str, Field(description="BIDS subject ID")]
    session_id: Annotated[
        Optional[str], Field(description="Session ID, e.g., ses-01")] = None
    task_id: Optional[str] = None
    run_id: Optional[str] = None
    pipeline: Annotated[
        str, Field(description="Pipeline name and version, e.g., freesurfer")
    ]
    complete_timestamp: Annotated[
        Optional[str], Field(description="Completion date")
    ] = None
    rater: Annotated[Optional[str], Field(description="Name of the rater")] = None
    metrics: Annotated[
        List[MetricQC], Field(description="List of metrics/criteria for this subject")
    ]
    require_rerun: Optional[str] = None
    final_qc: Optional[str] = None
