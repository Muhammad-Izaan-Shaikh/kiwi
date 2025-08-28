# === pages/5_Reporting.py ===
"""
Reporting & export page
Save as: pages/5_Reporting.py
"""
import streamlit as st
from utils import reporting, fileio, viz

st.set_page_config(page_title="5 - Reporting", layout="wide")
st.title("📝 Reporting & Export")

if "corr_df" not in st.session_state and "mlr_results" not in st.session_state:
    st.info("Run EDA and Modeling to produce artifacts to export.")

# Collect artifacts
corr = st.session_state.get("corr_df")
pvals = st.session_state.get("pval_df")
mlr = st.session_state.get("mlr_results")
figs = {}
if corr is not None:
    figs["Correlation Heatmap"] = viz.create_correlation_heatmap(corr, pvals)
if mlr is not None:
    try:
        from utils import modeling, viz as _viz
        coef_tbl = modeling.coef_table(mlr)
        figs["Coef Plot"] = _viz.create_coefficient_plot(coef_tbl)
    except Exception:
        pass

# Export PDF report
out_name = st.text_input("Report filename", value="analysis_report.pdf")
if st.button("Generate PDF report"):
    with st.spinner("Building report..."):
        txt_summary = mlr["results"].summary().as_text() if mlr is not None else None
        reporting.export_report(out_name, corr_table=corr, reg_summary=txt_summary, figures=figs)
    st.success("Report generated")
    st.download_button("Download report", data=fileio.read_file_bytes(out_name), file_name=out_name)

# Bundle everything in a zip
if st.button("Build artifact bundle (zip)"):
    bundle_bytes = fileio.build_artifact_bundle(
        cleaned_df=st.session_state.get("clean_df"),
        corr_df=corr,
        pval_df=pvals,
        mlr_summary=mlr["results"].summary().as_text() if mlr is not None else None,
        figures=figs
    )
    st.download_button("Download bundle", data=bundle_bytes, file_name="analysis_artifacts.zip")
