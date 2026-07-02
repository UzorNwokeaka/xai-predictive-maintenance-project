import streamlit as st


def kpi_row(metrics):
    """
    Display a row of KPI metrics.

    Parameters
    ----------
    metrics : list of tuples
        Each tuple should contain:
        (label, value, delta)

        Example:
        [
            ("RMSE", "15.32", None),
            ("Best Model", "Improved LSTM", None)
        ]
    """

    cols = st.columns(len(metrics))

    for col, (label, value, delta) in zip(cols, metrics):
        with col:
            st.metric(
                label=label,
                value=value,
                delta=delta,
            )