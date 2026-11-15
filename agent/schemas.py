"""Pydantic response schemas for cross-system MLOps diagnostics."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

CheckStatus = Literal[
    "success",
    "partial",
    "failed",
    "unavailable",
    "skipped",
]

OverallStatus = Literal[
    "complete",
    "partial",
    "failed",
]


class ToolResult(BaseModel):
    """Preserve one subsystem's result independently of other checks."""

    model_config = ConfigDict(extra="forbid")

    status: CheckStatus
    data: dict[str, Any] | list[Any] | None = None
    error: str | None = None
    checked_at: datetime | None = None
    duration_ms: float | None = Field(default=None, ge=0)


class DiagnosticFinding(BaseModel):
    """A factual observation or limitation from the diagnostic process."""

    model_config = ConfigDict(extra="forbid")

    severity: Literal["info", "warning", "error"]
    source: str
    message: str


class DiagnosticReport(BaseModel):
    """Consolidated, machine-readable MLOps diagnostic report."""

    model_config = ConfigDict(extra="forbid")

    generated_at: datetime
    overall_status: OverallStatus

    model_selection: ToolResult
    training_provenance: ToolResult
    dataset_state: ToolResult
    airflow: ToolResult
    kubernetes: ToolResult
    inference_api: ToolResult

    findings: list[DiagnosticFinding] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
