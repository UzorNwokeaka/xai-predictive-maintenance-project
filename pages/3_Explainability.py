import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (
    APP_TITLE,
    PRIMARY_XAI_METHOD,
    SUPPLEMENTARY_XAI_METHOD,
    PREDICTIVE_MODEL_NAME,
)

from utils.data_loader import (
    load_dashboard_data,
    load_temporal_explainability,
    load_model_metadata,
    load_timeshap_feature_attribution,
    load_timeshap_event_attribution,
    load_timeshap_pruning,
    load_ig_feature_attribution,
    load_ig_event_attribution,
    load_ig_sensor_time_attribution,
    load_xai_feature_deletion_fidelity,
    load_xai_method_scorecard,
    load_xai_comparison_summary,
    load_selected_features,
)

from utils.layout import render_page, end_page
from utils.helpers import section_title


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title=f"Explainability | {APP_TITLE}",
    layout="wide",
)

render_page(
    "Explainable AI Dashboard",
    (
        "Understand the feature, temporal and sensor-time factors "
        "influencing Remaining Useful Life predictions generated "
        "by the Improved LSTM."
    ),
)


# ============================================================
# Load application artefacts
# ============================================================

dashboard_df = load_dashboard_data()
temporal_df = load_temporal_explainability()
metadata = load_model_metadata()

timeshap_feature_df = load_timeshap_feature_attribution()
timeshap_event_df = load_timeshap_event_attribution()
timeshap_pruning_df = load_timeshap_pruning()

ig_feature_df = load_ig_feature_attribution()
ig_event_df = load_ig_event_attribution()
ig_sensor_time = load_ig_sensor_time_attribution()

fidelity_df = load_xai_feature_deletion_fidelity()
scorecard_df = load_xai_method_scorecard()
comparison_summary = load_xai_comparison_summary()

selected_features = list(load_selected_features())


# ============================================================
# Data preparation helpers
# ============================================================

def normalise_feature_attribution(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Standardise attribution files exported from Kaggle.

    Expected output:
        feature
        attribution
        absolute_attribution
    """
    df = dataframe.copy()

    rename_map = {}

    for column in df.columns:
        lower_column = column.lower().strip()

        if lower_column in {"feature", "variable"}:
            rename_map[column] = "feature"

        elif lower_column in {
            "attribution",
            "shapley value",
            "shapley_value",
            "shap value",
            "shap_value",
        }:
            rename_map[column] = "attribution"

        elif lower_column in {
            "absolute_attribution",
            "absolute attribution",
        }:
            rename_map[column] = "absolute_attribution"

    df = df.rename(columns=rename_map)

    required_columns = {"feature", "attribution"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "Feature attribution file must contain feature "
            "and attribution columns."
        )

    if "absolute_attribution" not in df.columns:
        df["absolute_attribution"] = df["attribution"].abs()

    return df[
        [
            "feature",
            "attribution",
            "absolute_attribution",
        ]
    ].copy()


def normalise_event_attribution(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Standardise TimeSHAP or Integrated Gradients event data.
    """
    df = dataframe.copy()

    rename_map = {}

    for column in df.columns:
        lower_column = column.lower().strip()

        if lower_column in {
            "event",
            "feature",
            "relative_timestep",
            "cycle",
        }:
            rename_map[column] = lower_column

        elif lower_column in {
            "attribution",
            "shapley value",
            "shapley_value",
            "shap value",
            "shap_value",
        }:
            rename_map[column] = "attribution"

        elif lower_column in {
            "absolute_attribution",
            "absolute attribution",
        }:
            rename_map[column] = "absolute_attribution"

    df = df.rename(columns=rename_map)

    if "attribution" not in df.columns:
        raise ValueError(
            "Event attribution file does not contain "
            "an attribution column."
        )

    if "absolute_attribution" not in df.columns:
        df["absolute_attribution"] = df["attribution"].abs()

    return df


def get_summary_value(
    key: str,
    default=None,
):
    """
    Safely retrieve a value from the XAI comparison summary.
    """
    return comparison_summary.get(key, default)


timeshap_feature_df = normalise_feature_attribution(
    timeshap_feature_df
)

ig_feature_df = normalise_feature_attribution(
    ig_feature_df
)

timeshap_event_df = normalise_event_attribution(
    timeshap_event_df
)

ig_event_df = normalise_event_attribution(
    ig_event_df
)


# ============================================================
# 1. Explainability overview
# ============================================================

section_title("1. Direct LSTM Explainability Overview")

st.markdown(
    """
    The final Improved LSTM is explained directly using two
    complementary post-hoc Explainable Artificial Intelligence
    techniques. TimeSHAP provides sequence-specific feature,
    event and coalition explanations, while Integrated Gradients
    provides independent feature, timestep and sensor-time
    attribution.
    """
)

overview_col1, overview_col2, overview_col3, overview_col4 = (
    st.columns(4)
)

overview_col1.metric(
    "Predictive Model",
    PREDICTIVE_MODEL_NAME,
)

overview_col2.metric(
    "Primary XAI Method",
    PRIMARY_XAI_METHOD,
)

overview_col3.metric(
    "Supplementary Method",
    SUPPLEMENTARY_XAI_METHOD,
)

overview_col4.metric(
    "Dataset",
    metadata.get(
        "dataset",
        "NASA C-MAPSS FD001",
    ),
)

st.info(
    """
    Both explanation methods operate directly on the same
    Improved LSTM used for Remaining Useful Life prediction.
    The Tuned Random Forest is retained only as an interpretable
    predictive baseline within the Model Comparison module.
    """
)

st.divider()


# ============================================================
# 2. TimeSHAP feature explanation
# ============================================================

section_title("2. TimeSHAP Feature Explanation")

st.markdown(
    """
    TimeSHAP is the primary explanation method because it was
    developed specifically for recurrent sequence models. The
    chart below identifies the sensor and operational variables
    that most strongly influenced the representative LSTM
    prediction.
    """
)

max_timeshap_features = len(timeshap_feature_df)

timeshap_top_n = st.slider(
    "Number of TimeSHAP features to display",
    min_value=5,
    max_value=max_timeshap_features,
    value=min(10, max_timeshap_features),
    key="timeshap_feature_count",
)

timeshap_top_df = (
    timeshap_feature_df
    .sort_values(
        "absolute_attribution",
        ascending=False,
    )
    .head(timeshap_top_n)
)

timeshap_feature_chart = px.bar(
    timeshap_top_df.sort_values("attribution"),
    x="attribution",
    y="feature",
    orientation="h",
    title="TimeSHAP Feature Attribution",
    labels={
        "attribution": "Contribution to predicted RUL",
        "feature": "Feature",
    },
)

timeshap_feature_chart.add_vline(
    x=0,
    line_width=1,
)

st.plotly_chart(
    timeshap_feature_chart,
    use_container_width=True,
)

st.dataframe(
    timeshap_top_df,
    use_container_width=True,
    hide_index=True,
)

top_timeshap_features = (
    timeshap_top_df["feature"]
    .head(5)
    .tolist()
)

st.info(
    f"""
    **Engineering interpretation**

    The strongest TimeSHAP degradation indicators are
    **{", ".join(top_timeshap_features)}**.

    Negative attributions reduce the predicted RUL relative to
    the healthy operating baseline, while positive attributions
    increase it.
    """
)

st.divider()


# ============================================================
# 3. TimeSHAP event and pruning explanation
# ============================================================

section_title("3. TimeSHAP Event and Coalition Explanation")

st.markdown(
    """
    Event attribution indicates how individual observations within
    the 30-cycle sequence influenced the LSTM prediction. Coalition
    pruning evaluates whether earlier events can be removed without
    materially changing the explanation.
    """
)

timeshap_event_plot_df = timeshap_event_df.copy()

event_label_column = None

for candidate in [
    "event",
    "feature",
    "relative_timestep",
    "cycle",
]:
    if candidate in timeshap_event_plot_df.columns:
        event_label_column = candidate
        break

if event_label_column is None:
    timeshap_event_plot_df[
        "event_position"
    ] = np.arange(
        1,
        len(timeshap_event_plot_df) + 1,
    )

    event_label_column = "event_position"

timeshap_event_chart = px.bar(
    timeshap_event_plot_df,
    x=event_label_column,
    y="attribution",
    title="TimeSHAP Event Attribution Across the 30-Cycle Window",
    labels={
        event_label_column: "Event position",
        "attribution": "Contribution to predicted RUL",
    },
)

timeshap_event_chart.add_hline(
    y=0,
    line_width=1,
)

st.plotly_chart(
    timeshap_event_chart,
    use_container_width=True,
)

pruning_index = get_summary_value(
    "coalition_pruning_index",
    None,
)

if pruning_index is None and not timeshap_pruning_df.empty:
    pruning_message = (
        "The pruning analysis was successfully generated. "
        "Review the table below for coalition-level results."
    )
else:
    pruning_message = (
        f"The TimeSHAP coalition pruning index was "
        f"**{pruning_index}**."
    )

st.success(
    f"""
    **Coalition pruning interpretation**

    {pruning_message}

    A value of **−30** indicates that the complete 30-cycle
    observation window contributed to the explanation and that no
    historical events could be removed under the selected tolerance.
    """
)

with st.expander(
    "View TimeSHAP pruning data",
):
    st.dataframe(
        timeshap_pruning_df,
        use_container_width=True,
        hide_index=True,
    )

st.divider()


# ============================================================
# 4. Integrated Gradients validation
# ============================================================

section_title("4. Integrated Gradients Validation")

st.markdown(
    """
    Integrated Gradients provides an independent gradient-based
    explanation of the same Improved LSTM. Its feature and timestep
    attributions are used to validate whether TimeSHAP identifies
    consistent degradation indicators.
    """
)

ig_col1, ig_col2 = st.columns(2)

with ig_col1:
    ig_top_df = (
        ig_feature_df
        .sort_values(
            "absolute_attribution",
            ascending=False,
        )
        .head(10)
    )

    ig_feature_chart = px.bar(
        ig_top_df.sort_values("attribution"),
        x="attribution",
        y="feature",
        orientation="h",
        title="Integrated Gradients Feature Attribution",
        labels={
            "attribution": "Contribution to predicted RUL",
            "feature": "Feature",
        },
    )

    ig_feature_chart.add_vline(
        x=0,
        line_width=1,
    )

    st.plotly_chart(
        ig_feature_chart,
        use_container_width=True,
    )

with ig_col2:
    ig_event_plot_df = ig_event_df.copy()

    if "cycle" in ig_event_plot_df.columns:
        ig_x_column = "cycle"
        ig_x_label = "Operational cycle"

    elif "relative_timestep" in ig_event_plot_df.columns:
        ig_x_column = "relative_timestep"
        ig_x_label = "Relative timestep"

    else:
        ig_event_plot_df[
            "event_position"
        ] = np.arange(
            1,
            len(ig_event_plot_df) + 1,
        )

        ig_x_column = "event_position"
        ig_x_label = "Event position"

    ig_event_chart = px.bar(
        ig_event_plot_df,
        x=ig_x_column,
        y="attribution",
        title="Integrated Gradients Event Attribution",
        labels={
            ig_x_column: ig_x_label,
            "attribution": "Contribution to predicted RUL",
        },
    )

    ig_event_chart.add_hline(
        y=0,
        line_width=1,
    )

    st.plotly_chart(
        ig_event_chart,
        use_container_width=True,
    )

st.divider()


# ============================================================
# 5. Sensor-time attribution heatmap
# ============================================================

section_title("5. Sensor-Time Attribution Heatmap")

st.markdown(
    """
    The sensor-time heatmap shows how each selected feature
    influenced the LSTM prediction at every timestep within the
    30-cycle input window. This reveals both which sensors mattered
    and when their influence became strongest.
    """
)

if ig_sensor_time.shape != (
    30,
    len(selected_features),
):
    st.warning(
        (
            "The sensor-time matrix shape does not exactly match "
            "the expected 30 × feature configuration. The heatmap "
            "will use the available dimensions."
        )
    )

heatmap_features = selected_features[
    : ig_sensor_time.shape[1]
]

heatmap_timesteps = list(
    range(
        -ig_sensor_time.shape[0] + 1,
        1,
    )
)

heatmap_figure = go.Figure(
    data=go.Heatmap(
        z=ig_sensor_time.T,
        x=heatmap_timesteps,
        y=heatmap_features,
        colorbar={
            "title": "Attribution",
        },
        hovertemplate=(
            "Feature: %{y}<br>"
            "Relative timestep: %{x}<br>"
            "Attribution: %{z:.4f}"
            "<extra></extra>"
        ),
    )
)

heatmap_figure.update_layout(
    title=(
        "Integrated Gradients Sensor-Time Attribution "
        "for the 30-Cycle Window"
    ),
    xaxis_title=(
        "Relative timestep "
        "(0 = most recent observation)"
    ),
    yaxis_title="Feature",
    height=650,
)

st.plotly_chart(
    heatmap_figure,
    use_container_width=True,
)

critical_start = get_summary_value(
    "critical_start_cycle",
    None,
)

critical_end = get_summary_value(
    "critical_end_cycle",
    None,
)

if (
    critical_start is not None
    and critical_end is not None
):
    st.warning(
        f"""
        **Critical degradation window**

        The highest cumulative temporal attribution occurred between
        operational cycles **{critical_start} and {critical_end}**.

        This interval represents the period that contributed most
        strongly to the representative Critical engine prediction.
        """
    )

st.divider()


# ============================================================
# 6. Explainability faithfulness and agreement
# ============================================================

section_title("6. Explanation Faithfulness and Agreement")

st.markdown(
    """
    Explanation quality is evaluated through feature deletion and
    cross-method ranking agreement. Masking highly ranked variables
    with healthy-reference values should materially change the LSTM
    prediction if the explanations are faithful.
    """
)

fidelity_chart = px.line(
    fidelity_df,
    x="fraction_masked",
    y="absolute_prediction_change",
    color="method",
    markers=True,
    title="Feature-Deletion Faithfulness Test",
    labels={
        "fraction_masked": "Fraction of top features masked",
        "absolute_prediction_change": (
            "Absolute change in predicted RUL"
        ),
        "method": "XAI method",
    },
)

st.plotly_chart(
    fidelity_chart,
    use_container_width=True,
)

agreement_col1, agreement_col2, agreement_col3 = (
    st.columns(3)
)

rank_correlation = get_summary_value(
    "feature_rank_spearman",
    0,
)

top_five_overlap = get_summary_value(
    "top_five_feature_overlap",
    0,
)

baseline_correlation = get_summary_value(
    "ig_baseline_rank_spearman",
    0,
)

agreement_col1.metric(
    "Feature-Rank Correlation",
    f"{rank_correlation:.4f}",
)

agreement_col2.metric(
    "Top-Five Feature Overlap",
    f"{top_five_overlap:.0%}",
)

agreement_col3.metric(
    "IG Baseline Rank Correlation",
    f"{baseline_correlation:.4f}",
)

st.info(
    """
    The very high TimeSHAP–Integrated Gradients rank correlation
    demonstrates strong agreement between two theoretically
    different explanation methods. The lower baseline correlation
    confirms that Integrated Gradients explanations depend on the
    reference state and supports the use of a physically meaningful
    healthy baseline rather than an all-zero sequence.
    """
)

with st.expander(
    "View XAI method scorecard",
):
    st.dataframe(
        scorecard_df,
        use_container_width=True,
        hide_index=True,
    )

st.divider()


# ============================================================
# 7. Local engine-level decision explanation
# ============================================================

section_title("7. Local Engine-Level Decision Explanation")

st.markdown(
    """
    The engine-level panel links each predicted RUL to its health
    status, risk level and recommended maintenance action.
    """
)

engine_ids = (
    dashboard_df["engine_id"]
    .sort_values()
    .unique()
)

selected_engine = st.selectbox(
    "Select Engine ID",
    engine_ids,
    key="local_engine_selector",
)

engine_record = dashboard_df[
    dashboard_df["engine_id"] == selected_engine
].iloc[0]

local_col1, local_col2, local_col3, local_col4 = (
    st.columns(4)
)

local_col1.metric(
    "Actual RUL",
    f"{engine_record['actual_rul']:.2f} cycles",
)

local_col2.metric(
    "Predicted RUL",
    (
        f"{engine_record['predicted_rul_lstm']:.2f} "
        "cycles"
    ),
)

local_col3.metric(
    "Health Status",
    engine_record["health_status"],
)

local_col4.metric(
    "Risk Level",
    engine_record["risk_level"],
)

st.info(
    f"""
    **Engine {int(selected_engine)} decision explanation**

    The Improved LSTM predicts approximately
    **{engine_record['predicted_rul_lstm']:.2f} operational cycles**
    remaining.

    The engine is classified as
    **{engine_record['health_status']}**, with a
    **{engine_record['risk_level']}** level.

    Recommended action:
    **{engine_record['recommendation']}**
    """
)

st.divider()


# ============================================================
# 8. Descriptive temporal sensor trends
# ============================================================

section_title("8. Descriptive Temporal Sensor Trends")

st.markdown(
    """
    This section retains the original interactive sensor trend
    analysis as descriptive engineering context. Unlike TimeSHAP
    and Integrated Gradients, these plots visualise sensor behaviour
    but do not independently attribute the LSTM prediction.
    """
)

available_temporal_engines = (
    temporal_df["engine_id"]
    .sort_values()
    .unique()
)

selected_temporal_engine = st.selectbox(
    "Select engine for temporal trend analysis",
    available_temporal_engines,
    key="temporal_engine_selector",
)

engine_temporal_df = temporal_df[
    temporal_df["engine_id"]
    == selected_temporal_engine
].sort_values("cycle")

feature_columns = [
    column
    for column in engine_temporal_df.columns
    if column not in [
        "engine_id",
        "cycle",
    ]
]

selected_temporal_features = st.multiselect(
    "Select features to visualise over time",
    feature_columns,
    default=feature_columns[
        : min(
            3,
            len(feature_columns),
        )
    ],
)

if selected_temporal_features:
    temporal_long_df = engine_temporal_df.melt(
        id_vars=[
            "engine_id",
            "cycle",
        ],
        value_vars=selected_temporal_features,
        var_name="Feature",
        value_name="Scaled Value",
    )

    temporal_chart = px.line(
        temporal_long_df,
        x="cycle",
        y="Scaled Value",
        color="Feature",
        markers=True,
        title=(
            "Last Operational Window Trend - "
            f"Engine {int(selected_temporal_engine)}"
        ),
    )

    st.plotly_chart(
        temporal_chart,
        use_container_width=True,
    )

    trend_summaries = []

    for feature in selected_temporal_features:
        start_value = (
            engine_temporal_df[feature]
            .iloc[0]
        )

        end_value = (
            engine_temporal_df[feature]
            .iloc[-1]
        )

        change = end_value - start_value

        if change > 0.05:
            trend = "increased"

        elif change < -0.05:
            trend = "decreased"

        else:
            trend = "remained relatively stable"

        trend_summaries.append(
            f"{feature} {trend}"
        )

    st.info(
        f"""
        **Engineering interpretation**

        For Engine **{int(selected_temporal_engine)}**:

        - {"; ".join(trend_summaries)}.

        These descriptive trends provide operational context for
        the model-attribution results.
        """
    )

else:
    st.warning(
        "Select at least one feature to display temporal trends."
    )

st.divider()


# ============================================================
# 9. Explainability summary
# ============================================================

section_title("9. Explainability Summary")

summary_col1, summary_col2 = st.columns(2)

with summary_col1:
    st.metric(
        "Predictive Model",
        PREDICTIVE_MODEL_NAME,
    )

    st.metric(
        "Primary XAI Method",
        PRIMARY_XAI_METHOD,
    )

with summary_col2:
    st.metric(
        "Supplementary XAI Method",
        SUPPLEMENTARY_XAI_METHOD,
    )

    st.metric(
        "Explanation Types",
        (
            "Feature, Event, Coalition, "
            "Sensor-Time"
        ),
    )

final_decision = get_summary_value(
    "provisional_xai_decision",
    (
        "TimeSHAP primary; "
        "Integrated Gradients supplementary"
    ),
)

st.success(
    f"""
    **Final XAI decision**

    {final_decision}

    TimeSHAP is adopted as the primary sequence-aware explanation
    technique, while Integrated Gradients provides independent
    feature, temporal and sensor-time validation. Together, both
    methods directly explain the deployed Improved LSTM and support
    transparent, human-centred predictive maintenance decisions.
    """
)

end_page()