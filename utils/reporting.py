import os
import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import TableStyle
from io import BytesIO
from reportlab.lib.pagesizes import A4

def correlation_table_to_pdf(corr, filename="psychology_report.pdf"):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("Psychology Data Analysis Report", styles['Title']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Correlation Matrix", styles['Heading2']))
    story.append(Spacer(1, 12))

    # Convert correlation DataFrame to list of lists
    data = [ ["" ] + list(corr.columns) ]  # header row
    for idx, row in corr.iterrows():
        data.append([idx] + [f"{val:.2f}" for val in row])

    # Create table
    table = Table(data, repeatRows=1)  # repeat header if table spans pages

    # Add style
    table.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("ALIGN", (1,1), (-1,-1), "CENTER"),
        ("FONTSIZE", (0,0), (-1,-1), 8),  # small font for fit
    ]))
    story.append(table)
    story.append(Spacer(1, 12))

    doc.build(story)
    print(f"✅ PDF saved as {filename}")

def export_report(filename: str, corr_table: pd.DataFrame = None, 
                  reg_summary: str = None, figures: dict = None):
    """
    Export analysis report as PDF
    
    Args:
        filename: Output PDF filename
        corr_table: Correlation table DataFrame
        reg_summary: Regression summary text (statsmodels output)
        figures: Dict of {title: matplotlib Figure}
    """
    doc = SimpleDocTemplate(filename, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []
    
    elements.append(Paragraph("Psychology Data Analysis Report", styles['Title']))
    elements.append(Spacer(1, 12))
    
    # Correlation table
    if corr_table is not None:
        elements.append(Paragraph("Correlation Matrix", styles['Heading2']))

        # Build table data with row + col labels
        data = [ [""] + list(corr_table.columns) ]  # header row
        for idx, row in corr_table.iterrows():
            data.append([idx] + [f"{val:.2f}" for val in row])

        # Set column widths to fit within page
        max_cols = len(data[0])
        col_width = (A4[0] - 100) / max_cols   # 100 for left+right margins
        col_widths = [col_width] * max_cols

        table = Table(data, colWidths=col_widths, repeatRows=1)

        # Add style
        table.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("ALIGN", (1,1), (-1,-1), "CENTER"),
            ("FONTSIZE", (0,0), (-1,-1), 6),   # smaller font for big tables
        ]))

        elements.append(table)
        elements.append(Spacer(1, 12))

    # Regression summary
    if reg_summary is not None:
        elements.append(Paragraph("Regression Analysis", styles['Heading2']))
        for line in reg_summary.split("\n"):
            elements.append(Paragraph(line, styles['Normal']))
        elements.append(Spacer(1, 12))
    
    # Figures
    if figures is not None:
        for title, fig in figures.items():
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            elements.append(Paragraph(title, styles['Heading2']))
            elements.append(Image(buf, width=400, height=300))
            elements.append(Spacer(1, 12))
    
    doc.build(elements)
