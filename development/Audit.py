import streamlit as st

from config import APP_TITLE
from utils.audit import (
    run_file_dependency_audit,
    run_dataset_audit,
    run_quality_checks,
)


st.set_page_config(
    page_title=f"Audit | {APP_TITLE}",
    layout="wide",
)

st.title("Application Audit and Validation")

st.markdown(
    """
    This page validates file dependencies, dataset structure, and core application
    quality checks before final deployment and dissertation documentation.
    """
)

st.subheader("File Dependency Audit")

file_audit_df = run_file_dependency_audit()
st.dataframe(file_audit_df, use_container_width=True)

if file_audit_df["exists"].all():
    st.success("All required files are present.")
else:
    st.error("Some required files are missing.")

st.divider()

st.subheader("Dataset and Metadata Audit")

dataset_audit = run_dataset_audit()

st.json(dataset_audit)

st.divider()

st.subheader("Quality Checks")

quality_df = run_quality_checks()
st.dataframe(quality_df, use_container_width=True)

if quality_df["status"].all():
    st.success("All quality checks passed.")
else:
    st.warning("Some quality checks require attention.")