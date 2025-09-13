# utils/ks_test.py

import pandas as pd
import numpy as np
from scipy.stats import kstest, norm
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def run_ks_test(df: pd.DataFrame, column: str, distribution="norm"):
    """
    Runs Kolmogorov-Smirnov test for a single numeric column
    against a specified theoretical distribution (default = normal).
    """

    data = df[column].dropna().values

    if len(data) == 0:
        return {"error": "No data available for this column."}

    # Default: normal distribution with sample mean and std
    if distribution == "norm":
        mean, std = np.mean(data), np.std(data, ddof=1)
        stat, pval = kstest(data, "norm", args=(mean, std))
    else:
        stat, pval = kstest(data, distribution)

    result = {
        "column": column,
        "distribution": distribution,
        "ks_statistic": stat,
        "p_value": pval,
        "conclusion": (
            "Fail to reject H0 (sample matches distribution)"
            if pval > 0.05
            else "Reject H0 (sample does not match distribution)"
        ),
    }

    return result


def export_ks_to_pdf(results: list, filename="ks_test_results.pdf"):
    """
    Exports a list of KS test results into a PDF file.
    """

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(200, height - 50, "Kolmogorov-Smirnov Test Results")

    y = height - 100
    c.setFont("Helvetica", 12)

    for res in results:
        if "error" in res:
            c.drawString(50, y, f"Column: {res.get('column', 'N/A')} - ERROR: {res['error']}")
        else:
            c.drawString(50, y, f"Column: {res['column']}")
            y -= 20
            c.drawString(70, y, f"Distribution: {res['distribution']}")
            y -= 20
            c.drawString(70, y, f"KS Statistic: {res['ks_statistic']:.4f}")
            y -= 20
            c.drawString(70, y, f"P-value: {res['p_value']:.4f}")
            y -= 20
            c.drawString(70, y, f"Conclusion: {res['conclusion']}")
        y -= 40

        if y < 100:
            c.showPage()
            y = height - 100

    c.save()
    buffer.seek(0)

    return buffer
