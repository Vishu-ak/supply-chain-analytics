"""
PDF procurement report generator.

Produces a professionally formatted executive report containing:
  - KPI scorecard with traffic-light status
  - Spend analysis summary with top vendors/categories
  - Vendor performance tier summary
  - Demand forecast summary
  - Inventory optimization recommendations
  - Risk flags and action items

Uses ReportLab for PDF generation with a branded layout.
"""

from __future__ import annotations

import io
import logging
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color palette (RGB 0-1 scale for ReportLab)
# ---------------------------------------------------------------------------
COLORS_RL = {
    "primary": (0.12, 0.23, 0.37),       # #1e3a5f
    "secondary": (0.16, 0.50, 0.73),     # #2980b9
    "accent": (0.15, 0.68, 0.38),        # #27ae60
    "warning": (0.90, 0.49, 0.13),       # #e67e22
    "danger": (0.91, 0.30, 0.24),        # #e74c3c
    "neutral": (0.58, 0.65, 0.65),       # #95a5a6
    "light_bg": (0.97, 0.97, 0.97),
    "white": (1.0, 1.0, 1.0),
    "black": (0.0, 0.0, 0.0),
    "dark_gray": (0.2, 0.2, 0.2),
}

STATUS_COLORS = {
    "on_target": COLORS_RL["accent"],
    "at_risk": COLORS_RL["warning"],
    "off_target": COLORS_RL["danger"],
    "unknown": COLORS_RL["neutral"],
}


class ProcurementPDFReport:
    """
    Generates a multi-page executive procurement report as a PDF file.

    Parameters
    ----------
    company_name : organization name displayed on cover page
    report_title : custom report title (default: "Procurement Analytics Report")
    logo_path : optional path to a PNG/JPEG logo file
    currency : currency symbol for monetary values

    Example
    -------
    >>> report = ProcurementPDFReport(company_name="Acme Corp")
    >>> report.add_kpi_section(kpi_df)
    >>> report.add_spend_section(spend_summary)
    >>> report.save("reports/Q4_2024_procurement.pdf")
    """

    def __init__(
        self,
        company_name: str = "Acme Corp",
        report_title: str = "Procurement Analytics Report",
        logo_path: Path | str | None = None,
        currency: str = "$",
    ) -> None:
        self.company_name = company_name
        self.report_title = report_title
        self.logo_path = Path(logo_path) if logo_path else None
        self.currency = currency
        self._sections: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def add_kpi_section(self, kpi_df: pd.DataFrame) -> "ProcurementPDFReport":
        """
        Add the KPI scorecard section.

        Parameters
        ----------
        kpi_df : output from KPITracker.to_dataframe()
        """
        self._sections.append({"type": "kpi", "data": kpi_df})
        return self

    def add_spend_section(
        self,
        spend_summary: dict[str, Any],
        top_vendors: pd.DataFrame | None = None,
        top_categories: pd.DataFrame | None = None,
    ) -> "ProcurementPDFReport":
        """Add the spend analysis section."""
        self._sections.append({
            "type": "spend",
            "summary": spend_summary,
            "top_vendors": top_vendors,
            "top_categories": top_categories,
        })
        return self

    def add_vendor_section(self, tier_summary: pd.DataFrame) -> "ProcurementPDFReport":
        """Add vendor performance tier summary."""
        self._sections.append({"type": "vendor", "data": tier_summary})
        return self

    def add_inventory_section(
        self, inventory_kpis: dict[str, Any], abc_summary: pd.DataFrame | None = None
    ) -> "ProcurementPDFReport":
        """Add inventory optimization section."""
        self._sections.append({
            "type": "inventory",
            "kpis": inventory_kpis,
            "abc_summary": abc_summary,
        })
        return self

    def add_recommendations(self, recommendations: list[dict[str, str]]) -> "ProcurementPDFReport":
        """
        Add an action items / recommendations section.

        Parameters
        ----------
        recommendations : list of dicts with keys: priority, area, action, expected_impact
        """
        self._sections.append({"type": "recommendations", "items": recommendations})
        return self

    # ------------------------------------------------------------------
    # PDF generation
    # ------------------------------------------------------------------

    def save(self, output_path: str | Path) -> Path:
        """
        Generate and save the PDF report.

        Parameters
        ----------
        output_path : file path for the output PDF

        Returns
        -------
        Path to the generated PDF
        """
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                HRFlowable, PageBreak,
            )
        except ImportError as exc:
            raise ImportError("reportlab is required: pip install reportlab") from exc

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=1.0 * inch,
            bottomMargin=0.75 * inch,
        )

        styles = getSampleStyleSheet()
        story: list[Any] = []

        # Styles
        title_style = ParagraphStyle(
            "ReportTitle",
            parent=styles["Title"],
            fontSize=24,
            textColor=colors.Color(*COLORS_RL["primary"]),
            spaceAfter=6,
        )
        h1_style = ParagraphStyle(
            "H1",
            parent=styles["Heading1"],
            fontSize=16,
            textColor=colors.Color(*COLORS_RL["primary"]),
            spaceBefore=16,
            spaceAfter=6,
        )
        h2_style = ParagraphStyle(
            "H2",
            parent=styles["Heading2"],
            fontSize=12,
            textColor=colors.Color(*COLORS_RL["secondary"]),
            spaceBefore=10,
            spaceAfter=4,
        )
        body_style = ParagraphStyle(
            "Body",
            parent=styles["Normal"],
            fontSize=9,
            leading=14,
            spaceAfter=4,
        )

        # Cover page
        story.extend(self._build_cover(title_style, body_style, styles))

        # Sections
        for section in self._sections:
            section_type = section["type"]
            if section_type == "kpi":
                story.extend(self._build_kpi_section(section["data"], h1_style, h2_style, body_style, colors))
            elif section_type == "spend":
                story.extend(self._build_spend_section(section, h1_style, h2_style, body_style, colors))
            elif section_type == "vendor":
                story.extend(self._build_vendor_section(section["data"], h1_style, h2_style, body_style, colors))
            elif section_type == "inventory":
                story.extend(self._build_inventory_section(section, h1_style, h2_style, body_style, colors))
            elif section_type == "recommendations":
                story.extend(self._build_recommendations(section["items"], h1_style, body_style, colors))

        doc.build(story)
        logger.info("PDF report saved: %s (%d KB)", output_path, output_path.stat().st_size // 1024)
        return output_path

    # ------------------------------------------------------------------
    # Section builders (internal)
    # ------------------------------------------------------------------

    def _build_cover(self, title_style: Any, body_style: Any, styles: Any) -> list[Any]:
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        from reportlab.platypus import Paragraph, Spacer, HRFlowable

        elements: list[Any] = [
            Spacer(1, 1.5 * inch),
            Paragraph(self.company_name, styles["Heading2"]),
            Paragraph(self.report_title, title_style),
            Spacer(1, 0.2 * inch),
            HRFlowable(width="100%", thickness=2, color=colors.Color(*COLORS_RL["primary"])),
            Spacer(1, 0.2 * inch),
            Paragraph(f"Generated: {date.today().strftime('%B %d, %Y')}", body_style),
            Paragraph("Prepared by: Supply Chain Analytics Platform", body_style),
            Spacer(1, 2 * inch),
        ]
        return elements

    def _build_kpi_section(
        self, kpi_df: pd.DataFrame, h1: Any, h2: Any, body: Any, colors: Any
    ) -> list[Any]:
        from reportlab.lib.units import inch
        from reportlab.platypus import Paragraph, Spacer, Table, TableStyle, PageBreak

        elements: list[Any] = [
            Paragraph("KPI Scorecard", h1),
            Paragraph(
                "The following KPIs reflect procurement performance for the current period.",
                body,
            ),
            Spacer(1, 0.2 * inch),
        ]

        if kpi_df.empty:
            elements.append(Paragraph("No KPI data available.", body))
            return elements

        # Table header
        display_cols = ["kpi", "value", "unit", "target", "achievement_pct", "status"]
        available = [c for c in display_cols if c in kpi_df.columns]
        header = ["KPI", "Value", "Unit", "Target", "Achievement", "Status"][:len(available)]

        table_data = [header]
        for _, row in kpi_df[available].iterrows():
            table_data.append([str(row[c]) for c in available])

        col_widths = [2.5 * inch, 0.8 * inch, 0.5 * inch, 0.8 * inch, 0.9 * inch, 0.9 * inch]
        col_widths = col_widths[:len(available)]

        t = Table(table_data, colWidths=col_widths)
        style_cmds = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.Color(*COLORS_RL["primary"])),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.Color(*COLORS_RL["light_bg"]), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.Color(*COLORS_RL["neutral"])),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]

        # Color status cells
        if "status" in available:
            status_col = available.index("status")
            for i, (_, row) in enumerate(kpi_df[available].iterrows(), start=1):
                status = str(row.get("status", "unknown"))
                fill = STATUS_COLORS.get(status, COLORS_RL["neutral"])
                style_cmds.append(("BACKGROUND", (status_col, i), (status_col, i), colors.Color(*fill)))
                style_cmds.append(("TEXTCOLOR", (status_col, i), (status_col, i), colors.white))

        t.setStyle(TableStyle(style_cmds))
        elements.append(t)
        return elements

    def _build_spend_section(self, section: dict, h1: Any, h2: Any, body: Any, colors: Any) -> list[Any]:
        from reportlab.lib.units import inch
        from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

        elements: list[Any] = [Paragraph("Spend Analysis", h1)]
        summary = section.get("summary", {})
        if summary:
            elements.append(Paragraph(
                f"Total Spend: {self.currency}{summary.get('total_spend_usd', 0):,.0f}  |  "
                f"Total POs: {summary.get('total_po_count', 0):,}  |  "
                f"Active Vendors: {summary.get('active_vendors', 0)}  |  "
                f"Maverick Spend: {summary.get('maverick_spend_pct', 0):.1f}%",
                body,
            ))

        if section.get("top_vendors") is not None:
            elements.append(Paragraph("Top Vendors by Spend", h2))
            df = section["top_vendors"].head(10)
            self._append_df_table(elements, df, colors, max_rows=10)

        return elements

    def _build_vendor_section(self, tier_df: pd.DataFrame, h1: Any, h2: Any, body: Any, colors: Any) -> list[Any]:
        from reportlab.platypus import Paragraph

        elements: list[Any] = [Paragraph("Vendor Performance by Tier", h1)]
        self._append_df_table(elements, tier_df, colors)
        return elements

    def _build_inventory_section(self, section: dict, h1: Any, h2: Any, body: Any, colors: Any) -> list[Any]:
        from reportlab.platypus import Paragraph, Spacer
        from reportlab.lib.units import inch

        elements: list[Any] = [Paragraph("Inventory Optimization", h1)]
        kpis = section.get("kpis", {})
        if kpis:
            lines = [
                f"Total SKUs: {kpis.get('total_skus', 0):,}",
                f"A-class items: {kpis.get('abc_a_count', 0)} | B: {kpis.get('abc_b_count', 0)} | C: {kpis.get('abc_c_count', 0)}",
                f"Avg EOQ: {kpis.get('avg_eoq', 0):.0f} units",
                f"Avg Safety Stock: {kpis.get('avg_safety_stock', 0):.0f} units",
                f"Avg Inventory Turnover: {kpis.get('avg_inventory_turnover', 0):.1f}×",
                f"Total Annual Inventory Cost: {self.currency}{kpis.get('total_annual_inv_cost_usd', 0):,.0f}",
            ]
            for line in lines:
                elements.append(Paragraph(line, body))
            elements.append(Spacer(1, 0.1 * inch))
        return elements

    def _build_recommendations(self, items: list[dict], h1: Any, body: Any, colors: Any) -> list[Any]:
        from reportlab.platypus import Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.units import inch

        elements: list[Any] = [
            Paragraph("Recommendations & Action Items", h1),
        ]
        if not items:
            return elements

        table_data = [["Priority", "Area", "Action", "Expected Impact"]]
        for item in items:
            table_data.append([
                item.get("priority", ""),
                item.get("area", ""),
                item.get("action", ""),
                item.get("expected_impact", ""),
            ])

        col_widths = [0.7 * inch, 1.0 * inch, 3.5 * inch, 1.8 * inch]
        t = Table(table_data, colWidths=col_widths)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.Color(*COLORS_RL["primary"])),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.Color(*COLORS_RL["light_bg"]), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.Color(*COLORS_RL["neutral"])),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("WORDWRAP", (0, 0), (-1, -1), True),
        ]))
        elements.append(t)
        return elements

    @staticmethod
    def _append_df_table(
        elements: list, df: pd.DataFrame, colors: Any, max_rows: int = 15
    ) -> None:
        from reportlab.lib.units import inch
        from reportlab.platypus import Table, TableStyle, Spacer

        df = df.head(max_rows)
        if df.empty:
            return

        headers = [str(c)[:18] for c in df.columns]
        rows = [[str(v)[:25] for v in row] for _, row in df.iterrows()]
        data = [headers] + rows

        n_cols = len(headers)
        col_width = max(0.5 * inch, 7.0 * inch / n_cols)
        t = Table(data, colWidths=[col_width] * n_cols, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.Color(*COLORS_RL["secondary"])),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.Color(*COLORS_RL["light_bg"]), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.Color(*COLORS_RL["neutral"])),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 0.15 * inch))
