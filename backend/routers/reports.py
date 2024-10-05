from fastapi import APIRouter, HTTPException, Response
from backend.services.data_loader import load_reports, export_analysis_report, export_experiment_report

router = APIRouter(prefix="/api/reports", tags=["reports"])

@router.get("")
def list_reports():
    """List all available reports."""
    return load_reports()

@router.get("/{report_id}/export")
def export_report(report_id: str):
    """Generate and download a markdown report."""
    markdown_content = export_analysis_report(report_id) or export_experiment_report(report_id)
    if not markdown_content:
        raise HTTPException(status_code=404, detail="Report data not found")
        
    return Response(
        content=markdown_content,
        media_type="text/markdown",
        headers={
            "Content-Disposition": f"attachment; filename=DeepShieldX_Report_{report_id}.md"
        }
    )
