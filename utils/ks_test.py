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
    try:
        # Check if column exists
        if column not in df.columns:
            return {"error": f"Column '{column}' not found in dataset."}
        
        data = df[column].dropna().values
        
        if len(data) == 0:
            return {"error": "No data available for this column."}
        
        # Need at least 8 observations for reliable KS test
        if len(data) < 8:
            return {"error": "Insufficient data points. Need at least 8 observations for reliable KS test."}
        
        # Default: normal distribution with sample mean and std
        if distribution == "norm":
            mean, std = np.mean(data), np.std(data, ddof=1)
            # Handle case where std is 0 (constant data)
            if std == 0:
                return {"error": "Data has zero variance. Cannot perform normality test on constant values."}
            stat, pval = kstest(data, "norm", args=(mean, std))
        else:
            stat, pval = kstest(data, distribution)
        
        result = {
            "column": column,
            "distribution": distribution,
            "ks_statistic": float(stat),  # Ensure JSON serializable
            "p_value": float(pval),       # Ensure JSON serializable
            "sample_size": len(data),
            "sample_mean": float(np.mean(data)),
            "sample_std": float(np.std(data, ddof=1)),
            "conclusion": (
                "Fail to reject H0 (sample matches distribution)"
                if pval > 0.05
                else "Reject H0 (sample does not match distribution)"
            ),
        }
        return result
        
    except Exception as e:
        return {"error": f"Error running KS test: {str(e)}"}

def export_ks_to_pdf(results: list, filename="ks_test_results.pdf"):
    """
    Exports a list of KS test results into a PDF file.
    """
    try:
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        
        # Title
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, height - 50, "Kolmogorov-Smirnov Test Results")
        
        # Date
        from datetime import datetime
        c.setFont("Helvetica", 10)
        c.drawString(50, height - 70, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        y = height - 110
        c.setFont("Helvetica", 12)
        
        for i, res in enumerate(results, 1):
            # Check if we need a new page
            if y < 150:
                c.showPage()
                y = height - 50
            
            if "error" in res:
                c.setFont("Helvetica-Bold", 12)
                c.drawString(50, y, f"{i}. Column: {res.get('column', 'N/A')}")
                y -= 20
                c.setFont("Helvetica", 11)
                c.drawString(70, y, f"ERROR: {res['error']}")
                y -= 30
            else:
                # Column header
                c.setFont("Helvetica-Bold", 12)
                c.drawString(50, y, f"{i}. Column: {res['column']}")
                y -= 25
                
                c.setFont("Helvetica", 11)
                c.drawString(70, y, f"Distribution tested: {res['distribution']}")
                y -= 18
                c.drawString(70, y, f"Sample size: {res.get('sample_size', 'N/A')}")
                y -= 18
                c.drawString(70, y, f"Sample mean: {res.get('sample_mean', 0):.4f}")
                y -= 18
                c.drawString(70, y, f"Sample std: {res.get('sample_std', 0):.4f}")
                y -= 18
                c.drawString(70, y, f"KS Statistic: {res['ks_statistic']:.6f}")
                y -= 18
                c.drawString(70, y, f"P-value: {res['p_value']:.6f}")
                y -= 18
                
                # Conclusion with emphasis
                c.setFont("Helvetica-Bold", 11)
                conclusion_text = f"Conclusion: {res['conclusion']}"
                c.drawString(70, y, conclusion_text)
                y -= 35
        
        c.save()
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        # Return empty buffer if PDF generation fails
        buffer = BytesIO()
        return buffer