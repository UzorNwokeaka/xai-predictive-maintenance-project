import pandas as pd
import streamlit as st

from config import (
    APP_TITLE,
    PREDICTIVE_MODEL_NAME,
    PRIMARY_XAI_METHOD,
    SUPPLEMENTARY_XAI_METHOD,
    WINDOW_SIZE,
)

from utils.layout import (
    render_page,
    end_page,
)

from utils.helpers import section_title

from utils.data_loader import (
    load_lstm_model,
    load_scaler,
    load_selected_features,
)

from utils.prediction import (
    predict_rul,
    generate_decision_output,
)


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title=f"Prediction | {APP_TITLE}",
    layout="wide",
)

render_page(
    "AI Predictive Maintenance Decision Support",
    (
        "Upload recent multivariate engine sensor data to "
        "predict Remaining Useful Life and translate the "
        "result into health status, risk level and "
        "maintenance guidance."
    ),
)


# ============================================================
# Load deployed model artefacts
# ============================================================

lstm_model = load_lstm_model()
scaler = load_scaler()

selected_features = list(
    load_selected_features()
)


# ============================================================
# Session-state initialisation
# ============================================================

if "prediction_completed" not in st.session_state:
    st.session_state.prediction_completed = False

if "input_reset_counter" not in st.session_state:
    st.session_state.input_reset_counter = 0


def clear_prediction_page():
    """
    Reset uploaded input and prediction outputs.
    """

    st.session_state.prediction_completed = False
    st.session_state.input_reset_counter += 1

    keys_to_remove = [
        "prediction_rul",
        "prediction_health_status",
        "prediction_risk_level",
        "prediction_recommendation",
        "prediction_source_file",
        "prediction_engine_id",
        "prediction_start_cycle",
        "prediction_end_cycle",
        "prediction_sequence_rows",
        "is_representative_xai_case",
    ]

    for key in keys_to_remove:
        st.session_state.pop(
            key,
            None,
        )


reset_id = (
    st.session_state.input_reset_counter
)


# ============================================================
# Helper functions
# ============================================================

def create_csv_template(
    feature_names,
    window_size,
):
    """
    Create a CSV template containing exactly WINDOW_SIZE rows.

    engine_id and cycle are metadata only.
    The predictive model uses only selected_features.
    """

    rows = []

    for cycle in range(
        1,
        window_size + 1,
    ):

        row = {
            "engine_id": 1,
            "cycle": cycle,
        }

        for feature in feature_names:
            row[feature] = 0.0

        rows.append(row)

    template_df = pd.DataFrame(
        rows
    )

    return (
        template_df
        .to_csv(index=False)
        .encode("utf-8")
    )


def validate_uploaded_sequence(
    uploaded_df,
    required_features,
):
    """
    Validate an uploaded engine sequence.

    Requirements:
    - exactly WINDOW_SIZE rows;
    - all selected model features present;
    - numeric model inputs;
    - no missing values;
    - consecutive cycle order where cycle is supplied;
    - only one engine ID where engine_id is supplied.
    """

    if uploaded_df is None or uploaded_df.empty:
        raise ValueError(
            "The uploaded CSV file contains no data."
        )

    missing_features = [
        feature
        for feature in required_features
        if feature not in uploaded_df.columns
    ]

    if missing_features:
        raise ValueError(
            "The uploaded file is missing the following "
            "required model features: "
            + ", ".join(missing_features)
        )

    if len(uploaded_df) != WINDOW_SIZE:
        raise ValueError(
            f"The Improved LSTM requires exactly "
            f"{WINDOW_SIZE} consecutive operational cycles. "
            f"The uploaded file contains "
            f"{len(uploaded_df)} rows."
        )

    validated_df = uploaded_df.copy()

    # ========================================================
    # Validate engine ID
    # ========================================================

    if "engine_id" in validated_df.columns:

        validated_df["engine_id"] = pd.to_numeric(
            validated_df["engine_id"],
            errors="coerce",
        )

        if validated_df["engine_id"].isnull().any():
            raise ValueError(
                "The engine_id column contains missing "
                "or non-numeric values."
            )

        engine_ids = (
            validated_df["engine_id"]
            .dropna()
            .unique()
        )

        if len(engine_ids) > 1:
            raise ValueError(
                "The uploaded CSV must contain data "
                "for only one engine."
            )

    # ========================================================
    # Validate operational cycles
    # ========================================================

    if "cycle" in validated_df.columns:

        validated_df["cycle"] = pd.to_numeric(
            validated_df["cycle"],
            errors="coerce",
        )

        if validated_df["cycle"].isnull().any():
            raise ValueError(
                "The cycle column contains missing "
                "or non-numeric values."
            )

        validated_df = (
            validated_df
            .sort_values("cycle")
            .reset_index(drop=True)
        )

        cycle_values = (
            validated_df["cycle"]
            .astype(int)
            .tolist()
        )

        expected_cycles = list(
            range(
                cycle_values[0],
                cycle_values[0] + WINDOW_SIZE,
            )
        )

        if cycle_values != expected_cycles:
            raise ValueError(
                "The uploaded cycle values must represent "
                f"{WINDOW_SIZE} consecutive operational cycles."
            )

    # ========================================================
    # Validate model features
    # ========================================================

    for feature in required_features:

        validated_df[feature] = pd.to_numeric(
            validated_df[feature],
            errors="coerce",
        )

    if (
        validated_df[required_features]
        .isnull()
        .any()
        .any()
    ):

        invalid_features = (
            validated_df[required_features]
            .columns[
                validated_df[
                    required_features
                ]
                .isnull()
                .any()
            ]
            .tolist()
        )

        raise ValueError(
            "Missing or non-numeric values were detected in: "
            + ", ".join(invalid_features)
        )

    return validated_df


def run_prediction(
    validated_df,
    source_file,
):
    """
    Run LSTM inference and generate maintenance
    decision-support outputs.
    """

    predicted_rul = predict_rul(
        input_df=validated_df,
        lstm_model=lstm_model,
        scaler=scaler,
        selected_features=selected_features,
    )

    decision = generate_decision_output(
        predicted_rul
    )

    # ========================================================
    # Save prediction result
    # ========================================================

    st.session_state.prediction_completed = True

    st.session_state.prediction_rul = float(
        decision["predicted_rul"]
    )

    st.session_state.prediction_health_status = (
        decision["health_status"]
    )

    st.session_state.prediction_risk_level = (
        decision["risk_level"]
    )

    st.session_state.prediction_recommendation = (
        decision["recommendation"]
    )

    # ========================================================
    # Save prediction traceability
    # ========================================================

    st.session_state.prediction_source_file = (
        source_file
    )

    st.session_state.prediction_sequence_rows = (
        len(validated_df)
    )

    if "engine_id" in validated_df.columns:

        st.session_state.prediction_engine_id = int(
            validated_df[
                "engine_id"
            ].iloc[0]
        )

    else:

        st.session_state.prediction_engine_id = None

    if "cycle" in validated_df.columns:

        st.session_state.prediction_start_cycle = int(
            validated_df[
                "cycle"
            ].iloc[0]
        )

        st.session_state.prediction_end_cycle = int(
            validated_df[
                "cycle"
            ].iloc[-1]
        )

    else:

        st.session_state.prediction_start_cycle = None
        st.session_state.prediction_end_cycle = None

    # ========================================================
    # Identify representative XAI case
    # ========================================================

    st.session_state.is_representative_xai_case = (
        st.session_state.prediction_engine_id == 41
        and st.session_state.prediction_start_cycle == 94
        and st.session_state.prediction_end_cycle == 123
        and st.session_state.prediction_health_status
        == "Critical"
    )


# ============================================================
# 1. Predictive model overview
# ============================================================

section_title(
    "1. Predictive Model Overview"
)

overview_col1, overview_col2, overview_col3 = (
    st.columns(3)
)

overview_col1.metric(
    "Deployed Model",
    PREDICTIVE_MODEL_NAME,
)

overview_col2.metric(
    "Input Window",
    f"{WINDOW_SIZE} operational cycles",
)

overview_col3.metric(
    "Input Variables",
    f"{len(selected_features)} features",
)

st.info(
    f"""
    The deployed **{PREDICTIVE_MODEL_NAME}** was trained
    using multivariate sequences containing
    **{WINDOW_SIZE} consecutive operational cycles**
    from the NASA C-MAPSS FD001 dataset.

    Each operational cycle contains the same
    **{len(selected_features)} selected model features**.

    The uploaded CSV therefore represents a genuine
    temporal engine sequence rather than a single
    sensor observation.
    """
)


# ============================================================
# 2. Upload engine data
# ============================================================

section_title(
    "2. Upload Engine Operational Data"
)

st.markdown(
    f"""
    Upload a CSV file containing exactly
    **{WINDOW_SIZE} consecutive operational cycles**
    for a single engine.

    The file must include the
    **{len(selected_features)} model input features**.

    The optional `engine_id` and `cycle` columns are used
    for identification and sequence validation only.
    They are **not predictive model features**.
    """
)


# ============================================================
# CSV template
# ============================================================

st.download_button(
    label="Download 30-Cycle CSV Template",
    data=create_csv_template(
        selected_features,
        WINDOW_SIZE,
    ),
    file_name=(
        "engine_30cycle_input_template.csv"
    ),
    mime="text/csv",
)


# ============================================================
# Upload control
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Engine Sensor Sequence CSV",
    type=["csv"],
    key=(
        f"uploaded_engine_sequence_"
        f"{reset_id}"
    ),
    help=(
        f"Upload exactly {WINDOW_SIZE} consecutive "
        "operational cycles for one engine."
    ),
)


validated_df = None


# ============================================================
# Validate uploaded file
# ============================================================

if uploaded_file is not None:

    try:

        uploaded_df = pd.read_csv(
            uploaded_file
        )

        validated_df = (
            validate_uploaded_sequence(
                uploaded_df,
                selected_features,
            )
        )

        st.success(
            "Engine sequence successfully "
            "loaded and validated."
        )

        # ====================================================
        # Sequence information
        # ====================================================

        section_title(
            "3. Sequence Validation"
        )

        (
            validation_col1,
            validation_col2,
            validation_col3,
        ) = st.columns(3)

        engine_display = "Not supplied"

        if "engine_id" in validated_df.columns:

            engine_display = str(
                int(
                    validated_df[
                        "engine_id"
                    ].iloc[0]
                )
            )

        validation_col1.metric(
            "Engine ID",
            engine_display,
        )

        validation_col2.metric(
            "Operational Cycles",
            f"{len(validated_df)}",
        )

        validation_col3.metric(
            "Model Features",
            f"{len(selected_features)}",
        )

        if "cycle" in validated_df.columns:

            start_cycle = int(
                validated_df[
                    "cycle"
                ].iloc[0]
            )

            end_cycle = int(
                validated_df[
                    "cycle"
                ].iloc[-1]
            )

            st.caption(
                f"Validated cycle range: "
                f"{start_cycle}–{end_cycle}"
            )

        # ====================================================
        # Preview
        # ====================================================

        section_title(
            "4. Uploaded Sequence Preview"
        )

        preview_columns = []

        if "engine_id" in validated_df.columns:
            preview_columns.append(
                "engine_id"
            )

        if "cycle" in validated_df.columns:
            preview_columns.append(
                "cycle"
            )

        preview_columns.extend(
            selected_features
        )

        st.dataframe(
            validated_df[
                preview_columns
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            f"Sequence shape before model scaling: "
            f"{len(validated_df)} × "
            f"{len(selected_features)} model features."
        )

    except Exception as error:

        st.error(
            "The uploaded CSV could not be validated: "
            f"{error}"
        )

        validated_df = None


else:

    st.info(
        """
        Upload a valid engine sequence CSV file
        to enable Remaining Useful Life prediction.
        """
    )


# ============================================================
# 5. AI inference controls
# ============================================================

section_title(
    "5. AI Inference"
)

predict_col, clear_col = st.columns(2)

with predict_col:

    predict_clicked = st.button(
        "Predict Remaining Useful Life",
        use_container_width=True,
        type="primary",
        disabled=(
            validated_df is None
        ),
    )

with clear_col:

    clear_clicked = st.button(
        "Clear Uploaded Data",
        use_container_width=True,
    )


# ============================================================
# Clear
# ============================================================

if clear_clicked:

    clear_prediction_page()
    st.rerun()


# ============================================================
# Run prediction
# ============================================================

if predict_clicked:

    if validated_df is None:

        st.error(
            "No valid engine sequence is available "
            "for prediction."
        )

        st.stop()

    try:

        run_prediction(
            validated_df=validated_df,
            source_file=uploaded_file.name,
        )

    except (
        ValueError,
        TypeError,
    ) as error:

        st.error(
            "The prediction could not be completed "
            "because the input sequence was invalid: "
            f"{error}"
        )

        st.stop()

    except Exception as error:

        st.error(
            "An unexpected error occurred during "
            "model inference: "
            f"{error}"
        )

        st.stop()


# ============================================================
# 6. Prediction output
# ============================================================

if st.session_state.prediction_completed:

    st.divider()

    section_title(
        "6. Prediction and Health Assessment"
    )

    (
        result_col1,
        result_col2,
        result_col3,
    ) = st.columns(3)

    result_col1.metric(
        "Predicted RUL",
        (
            f"{st.session_state.prediction_rul:.2f} "
            "cycles"
        ),
    )

    result_col2.metric(
        "Health Status",
        st.session_state.prediction_health_status,
    )

    result_col3.metric(
        "Risk Level",
        st.session_state.prediction_risk_level,
    )


    # ========================================================
    # 7. Prediction traceability
    # ========================================================

    section_title(
        "7. Prediction Traceability"
    )

    (
        trace_col1,
        trace_col2,
        trace_col3,
    ) = st.columns(3)

    trace_col1.metric(
        "Input File",
        st.session_state.prediction_source_file,
    )

    engine_id_display = (
        st.session_state.prediction_engine_id
        if st.session_state.prediction_engine_id
        is not None
        else "Not supplied"
    )

    trace_col2.metric(
        "Engine ID",
        engine_id_display,
    )

    start_cycle = (
        st.session_state.prediction_start_cycle
    )

    end_cycle = (
        st.session_state.prediction_end_cycle
    )

    if (
        start_cycle is not None
        and end_cycle is not None
    ):

        cycle_range = (
            f"{start_cycle}"
            f"–"
            f"{end_cycle}"
        )

    else:

        cycle_range = "Not supplied"

    trace_col3.metric(
        "Cycle Window",
        cycle_range,
    )


    # ========================================================
    # 8. Maintenance recommendation
    # ========================================================

    section_title(
        "8. Maintenance Recommendation"
    )

    recommendation = (
        st.session_state.prediction_recommendation
    )

    risk_level = (
        st.session_state.prediction_risk_level
    )

    if risk_level == "High Risk":

        st.error(
            recommendation
        )

    elif risk_level == "Medium Risk":

        st.warning(
            recommendation
        )

    else:

        st.success(
            recommendation
        )

    st.markdown(
        f"""
        The **{PREDICTIVE_MODEL_NAME}** estimates that
        this engine has approximately
        **{st.session_state.prediction_rul:.2f}
        operational cycles** remaining.

        The prediction corresponds to a
        **{st.session_state.prediction_health_status}**
        health condition and a
        **{st.session_state.prediction_risk_level}**
        risk level.

        The recommendation translates the numerical
        model output into operational maintenance
        decision support. Final maintenance action
        should remain subject to engineering inspection,
        operating context and organisational procedures.
        """
    )


    # ========================================================
    # 9. Explainability reference
    # ========================================================

    section_title(
        "9. Explainability Reference"
    )

    xai_col1, xai_col2 = st.columns(2)

    with xai_col1:

        st.metric(
            "Primary XAI Method",
            PRIMARY_XAI_METHOD,
        )

        st.markdown(
            """
            TimeSHAP provides sequence-aware feature,
            event and temporal explanations directly
            for the Improved LSTM.
            """
        )

    with xai_col2:

        st.metric(
            "Complementary XAI Method",
            SUPPLEMENTARY_XAI_METHOD,
        )

        st.markdown(
            """
            Integrated Gradients provides an independent
            gradient-based feature attribution perspective
            for comparison with TimeSHAP.
            """
        )

    if st.session_state.get(
        "is_representative_xai_case",
        False,
    ):

        st.success(
            """
            **Representative XAI Case Detected**

            This prediction corresponds to the
            representative Critical Engine 41 case used
            during the validated Explainable AI analysis.

            The Explainability module contains the saved
            TimeSHAP and Integrated Gradients results for
            this representative engine sequence.
            """
        )

    else:

        st.info(
            """
            The Explainability module presents validated
            TimeSHAP and Integrated Gradients artefacts for
            the representative Critical Engine 41 case.

            The current prediction demonstrates the
            predictive and decision-support functionality
            of the deployed Improved LSTM, but the saved
            XAI artefacts should not be interpreted as a
            case-specific explanation of this current
            Healthy or Warning prediction.
            """
        )


# ============================================================
# Initial guidance
# ============================================================

else:

    st.divider()

    st.info(
        f"""
        Upload exactly **{WINDOW_SIZE} consecutive
        operational cycles** for one engine and then
        select **Predict Remaining Useful Life**.

        The application will scale the
        {len(selected_features)} selected variables,
        construct the LSTM input sequence and generate
        the RUL prediction, health classification,
        risk assessment and maintenance recommendation.
        """
    )


end_page()



# import pandas as pd
# import streamlit as st

# from config import (
#     APP_TITLE,
#     PREDICTIVE_MODEL_NAME,
#     PRIMARY_XAI_METHOD,
#     SUPPLEMENTARY_XAI_METHOD,
#     WINDOW_SIZE,
# )

# from utils.layout import (
#     render_page,
#     end_page,
# )

# from utils.helpers import section_title

# from utils.data_loader import (
#     load_lstm_model,
#     load_scaler,
#     load_selected_features,
# )

# from utils.prediction import (
#     predict_rul,
#     generate_decision_output,
# )


# # ============================================================
# # Page configuration
# # ============================================================

# st.set_page_config(
#     page_title=f"Prediction | {APP_TITLE}",
#     layout="wide",
# )

# render_page(
#     "AI Predictive Maintenance Decision Support",
#     (
#         "Upload recent multivariate engine sensor data to "
#         "predict Remaining Useful Life (RUL) and translate the "
#         "result into health status, risk level and "
#         "maintenance guidance."
#     ),
# )


# # ============================================================
# # Load deployed model artefacts
# # ============================================================

# lstm_model = load_lstm_model()
# scaler = load_scaler()
# selected_features = list(
#     load_selected_features()
# )


# # ============================================================
# # Session-state initialisation
# # ============================================================

# if "prediction_completed" not in st.session_state:
#     st.session_state.prediction_completed = False

# if "input_reset_counter" not in st.session_state:
#     st.session_state.input_reset_counter = 0


# def clear_prediction_page():
#     """
#     Reset the uploaded input and prediction outputs.
#     """

#     st.session_state.prediction_completed = False
#     st.session_state.input_reset_counter += 1

#     keys_to_remove = [
#         "prediction_rul",
#         "prediction_health_status",
#         "prediction_risk_level",
#         "prediction_recommendation",
#         "prediction_source_file",
#         "prediction_engine_id",
#         "prediction_start_cycle",
#         "prediction_end_cycle",
#         "prediction_sequence_rows",
#     ]

#     for key in keys_to_remove:
#         st.session_state.pop(
#             key,
#             None,
#         )


# reset_id = (
#     st.session_state.input_reset_counter
# )


# # ============================================================
# # Helper functions
# # ============================================================

# def create_csv_template(
#     feature_names,
#     window_size,
# ):
#     """
#     Create a CSV template containing exactly WINDOW_SIZE rows.

#     engine_id and cycle are metadata columns only.
#     The model itself uses only selected_features.
#     """

#     rows = []

#     for cycle in range(
#         1,
#         window_size + 1,
#     ):
#         row = {
#             "engine_id": 1,
#             "cycle": cycle,
#         }

#         for feature in feature_names:
#             row[feature] = 0.0

#         rows.append(row)

#     template_df = pd.DataFrame(rows)

#     return (
#         template_df
#         .to_csv(index=False)
#         .encode("utf-8")
#     )


# def validate_uploaded_sequence(
#     uploaded_df,
#     required_features,
# ):
#     """
#     Validate uploaded engine sequence.

#     Requirements:
#     - exactly WINDOW_SIZE rows;
#     - all selected model features present;
#     - numeric model inputs;
#     - no missing values;
#     - cycle order validated when cycle column exists;
#     - only one engine ID allowed when engine_id exists.
#     """

#     if uploaded_df is None or uploaded_df.empty:
#         raise ValueError(
#             "The uploaded CSV file contains no data."
#         )

#     missing_features = [
#         feature
#         for feature in required_features
#         if feature not in uploaded_df.columns
#     ]

#     if missing_features:
#         raise ValueError(
#             "The uploaded file is missing the following "
#             "required model features: "
#             + ", ".join(missing_features)
#         )

#     if len(uploaded_df) != WINDOW_SIZE:
#         raise ValueError(
#             f"The Improved LSTM requires exactly "
#             f"{WINDOW_SIZE} consecutive operational cycles. "
#             f"The uploaded file contains "
#             f"{len(uploaded_df)} rows."
#         )

#     validated_df = uploaded_df.copy()

#     # --------------------------------------------------------
#     # Validate metadata
#     # --------------------------------------------------------

#     if "engine_id" in validated_df.columns:

#         engine_ids = (
#             validated_df["engine_id"]
#             .dropna()
#             .unique()
#         )

#         if len(engine_ids) > 1:
#             raise ValueError(
#                 "The uploaded CSV must contain data "
#                 "for only one engine."
#             )

#     if "cycle" in validated_df.columns:

#         validated_df["cycle"] = pd.to_numeric(
#             validated_df["cycle"],
#             errors="coerce",
#         )

#         if validated_df["cycle"].isnull().any():
#             raise ValueError(
#                 "The cycle column contains missing "
#                 "or non-numeric values."
#             )

#         validated_df = (
#             validated_df
#             .sort_values("cycle")
#             .reset_index(drop=True)
#         )

#         cycle_values = (
#             validated_df["cycle"]
#             .astype(int)
#             .tolist()
#         )

#         expected_cycles = list(
#             range(
#                 cycle_values[0],
#                 cycle_values[0] + WINDOW_SIZE,
#             )
#         )

#         if cycle_values != expected_cycles:
#             raise ValueError(
#                 "The uploaded cycle values must represent "
#                 f"{WINDOW_SIZE} consecutive operational cycles."
#             )

#     # --------------------------------------------------------
#     # Validate model features
#     # --------------------------------------------------------

#     for feature in required_features:

#         validated_df[feature] = pd.to_numeric(
#             validated_df[feature],
#             errors="coerce",
#         )

#     if (
#         validated_df[required_features]
#         .isnull()
#         .any()
#         .any()
#     ):
#         invalid_features = (
#             validated_df[required_features]
#             .columns[
#                 validated_df[
#                     required_features
#                 ]
#                 .isnull()
#                 .any()
#             ]
#             .tolist()
#         )

#         raise ValueError(
#             "Missing or non-numeric values were detected in: "
#             + ", ".join(invalid_features)
#         )

#     return validated_df


# def run_prediction(
#     validated_df,
#     source_file,
# ):
#     """
#     Run the complete LSTM inference and
#     decision-support pipeline.
#     """

#     predicted_rul = predict_rul(
#         input_df=validated_df,
#         lstm_model=lstm_model,
#         scaler=scaler,
#         selected_features=selected_features,
#     )

#     decision = generate_decision_output(
#         predicted_rul
#     )

#     st.session_state.prediction_completed = True

#     st.session_state.prediction_rul = float(
#         decision["predicted_rul"]
#     )

#     st.session_state.prediction_health_status = (
#         decision["health_status"]
#     )

#     st.session_state.prediction_risk_level = (
#         decision["risk_level"]
#     )

#     st.session_state.prediction_recommendation = (
#         decision["recommendation"]
#     )

#     st.session_state.prediction_source_file = (
#         source_file
#     )

#     st.session_state.prediction_sequence_rows = (
#         len(validated_df)
#     )

#     if "engine_id" in validated_df.columns:

#         st.session_state.prediction_engine_id = (
#             int(
#                 validated_df[
#                     "engine_id"
#                 ].iloc[0]
#             )
#         )

#     else:

#         st.session_state.prediction_engine_id = (
#             "Not supplied"
#         )

#     if "cycle" in validated_df.columns:

#         st.session_state.prediction_start_cycle = (
#             int(
#                 validated_df[
#                     "cycle"
#                 ].iloc[0]
#             )
#         )

#         st.session_state.prediction_end_cycle = (
#             int(
#                 validated_df[
#                     "cycle"
#                 ].iloc[-1]
#             )
#         )

#     else:

#         st.session_state.prediction_start_cycle = (
#             "Not supplied"
#         )

#         st.session_state.prediction_end_cycle = (
#             "Not supplied"
#         )


# # ============================================================
# # 1. Predictive model overview
# # ============================================================

# section_title(
#     "1. Predictive Model Overview"
# )

# overview_col1, overview_col2, overview_col3 = (
#     st.columns(3)
# )

# overview_col1.metric(
#     "Deployed Model",
#     PREDICTIVE_MODEL_NAME,
# )

# overview_col2.metric(
#     "Input Window",
#     f"{WINDOW_SIZE} operational cycles",
# )

# overview_col3.metric(
#     "Input Variables",
#     f"{len(selected_features)} features",
# )

# st.info(
#     f"""
#     The deployed **{PREDICTIVE_MODEL_NAME}** was trained
#     using multivariate sequences containing
#     **{WINDOW_SIZE} consecutive operational cycles**
#     from the NASA C-MAPSS FD001 dataset.

#     Each operational cycle contains the same
#     **{len(selected_features)} selected model features**.

#     The uploaded CSV therefore represents a genuine
#     temporal sequence rather than a single sensor
#     observation.
#     """
# )


# # ============================================================
# # 2. Upload engine data
# # ============================================================

# section_title(
#     "2. Upload Engine Operational Data"
# )

# st.markdown(
#     f"""
#     Upload a CSV file containing exactly
#     **{WINDOW_SIZE} consecutive operational cycles**
#     for a single engine.

#     The file must include the
#     **{len(selected_features)} model input features**.

#     The optional `engine_id` and `cycle` columns are used
#     only for identification and sequence validation;
#     they are **not predictive model features**.
#     """
# )


# # ============================================================
# # CSV template
# # ============================================================

# st.download_button(
#     label="Download 30-Cycle CSV Template",
#     data=create_csv_template(
#         selected_features,
#         WINDOW_SIZE,
#     ),
#     file_name=(
#         "engine_30cycle_input_template.csv"
#     ),
#     mime="text/csv",
# )


# # ============================================================
# # Upload control
# # ============================================================

# uploaded_file = st.file_uploader(
#     "Upload Engine Sensor Sequence CSV",
#     type=["csv"],
#     key=(
#         f"uploaded_engine_sequence_"
#         f"{reset_id}"
#     ),
#     help=(
#         f"Upload exactly {WINDOW_SIZE} consecutive "
#         "operational cycles for one engine."
#     ),
# )


# validated_df = None


# # ============================================================
# # Validate uploaded file
# # ============================================================

# if uploaded_file is not None:

#     try:

#         uploaded_df = pd.read_csv(
#             uploaded_file
#         )

#         validated_df = (
#             validate_uploaded_sequence(
#                 uploaded_df,
#                 selected_features,
#             )
#         )

#         st.success(
#             "Engine sequence successfully "
#             "loaded and validated."
#         )

#         # ----------------------------------------------------
#         # Sequence information
#         # ----------------------------------------------------

#         section_title(
#             "3. Sequence Validation"
#         )

#         validation_col1, validation_col2, (
#             validation_col3
#         ) = st.columns(3)

#         engine_display = "Not supplied"

#         if "engine_id" in validated_df.columns:

#             engine_display = str(
#                 int(
#                     validated_df[
#                         "engine_id"
#                     ].iloc[0]
#                 )
#             )

#         validation_col1.metric(
#             "Engine ID",
#             engine_display,
#         )

#         validation_col2.metric(
#             "Operational Cycles",
#             f"{len(validated_df)}",
#         )

#         validation_col3.metric(
#             "Model Features",
#             f"{len(selected_features)}",
#         )

#         if "cycle" in validated_df.columns:

#             start_cycle = int(
#                 validated_df[
#                     "cycle"
#                 ].iloc[0]
#             )

#             end_cycle = int(
#                 validated_df[
#                     "cycle"
#                 ].iloc[-1]
#             )

#             st.caption(
#                 f"Validated cycle range: "
#                 f"{start_cycle}–{end_cycle}"
#             )

#         # ----------------------------------------------------
#         # Preview
#         # ----------------------------------------------------

#         section_title(
#             "4. Uploaded Sequence Preview"
#         )

#         preview_columns = []

#         if "engine_id" in validated_df.columns:
#             preview_columns.append(
#                 "engine_id"
#             )

#         if "cycle" in validated_df.columns:
#             preview_columns.append(
#                 "cycle"
#             )

#         preview_columns.extend(
#             selected_features
#         )

#         st.dataframe(
#             validated_df[
#                 preview_columns
#             ],
#             use_container_width=True,
#             hide_index=True,
#         )

#         st.caption(
#             f"Sequence shape before model scaling: "
#             f"{len(validated_df)} × "
#             f"{len(selected_features)} model features."
#         )

#     except Exception as error:

#         st.error(
#             "The uploaded CSV could not be validated: "
#             f"{error}"
#         )

#         validated_df = None


# else:

#     st.info(
#         """
#         Upload a valid engine sequence CSV file
#         to enable Remaining Useful Life prediction.
#         """
#     )


# # ============================================================
# # 5. AI inference controls
# # ============================================================

# section_title(
#     "5. AI Inference"
# )

# predict_col, clear_col = st.columns(2)

# with predict_col:

#     predict_clicked = st.button(
#         "Predict Remaining Useful Life",
#         use_container_width=True,
#         type="primary",
#         disabled=(
#             validated_df is None
#         ),
#     )

# with clear_col:

#     clear_clicked = st.button(
#         "Clear Uploaded Data",
#         use_container_width=True,
#     )


# # ============================================================
# # Clear
# # ============================================================

# if clear_clicked:

#     clear_prediction_page()
#     st.rerun()


# # ============================================================
# # Prediction
# # ============================================================

# if predict_clicked:

#     if validated_df is None:

#         st.error(
#             "No valid engine sequence is available "
#             "for prediction."
#         )

#         st.stop()

#     try:

#         run_prediction(
#             validated_df=validated_df,
#             source_file=uploaded_file.name,
#         )

#     except (
#         ValueError,
#         TypeError,
#     ) as error:

#         st.error(
#             "The prediction could not be completed "
#             "because the input sequence was invalid: "
#             f"{error}"
#         )

#         st.stop()

#     except Exception as error:

#         st.error(
#             "An unexpected error occurred during "
#             "model inference: "
#             f"{error}"
#         )

#         st.stop()


# # ============================================================
# # 6. Prediction output
# # ============================================================

# if st.session_state.prediction_completed:

#     st.divider()

#     section_title(
#         "6. Prediction and Health Assessment"
#     )

#     result_col1, result_col2, result_col3 = (
#         st.columns(3)
#     )

#     result_col1.metric(
#         "Predicted RUL",
#         (
#             f"{st.session_state.prediction_rul:.2f} "
#             "cycles"
#         ),
#     )

#     result_col2.metric(
#         "Health Status",
#         st.session_state.prediction_health_status,
#     )

#     result_col3.metric(
#         "Risk Level",
#         st.session_state.prediction_risk_level,
#     )


#     # ========================================================
#     # Prediction traceability
#     # ========================================================

#     section_title(
#         "7. Prediction Traceability"
#     )

#     trace_col1, trace_col2, trace_col3 = (
#         st.columns(3)
#     )

#     trace_col1.metric(
#         "Input File",
#         st.session_state.prediction_source_file,
#     )

#     trace_col2.metric(
#         "Engine ID",
#         st.session_state.prediction_engine_id,
#     )

#     if (
#         st.session_state.prediction_start_cycle
#         != "Not supplied"
#     ):

#         cycle_range = (
#             f"{st.session_state.prediction_start_cycle}"
#             f"–"
#             f"{st.session_state.prediction_end_cycle}"
#         )

#     else:

#         cycle_range = "Not supplied"

#     trace_col3.metric(
#         "Cycle Window",
#         cycle_range,
#     )


#     # ========================================================
#     # Maintenance recommendation
#     # ========================================================

#     section_title(
#         "8. Maintenance Recommendation"
#     )

#     recommendation = (
#         st.session_state.prediction_recommendation
#     )

#     risk_level = (
#         st.session_state.prediction_risk_level
#     )

#     if risk_level == "High Risk":

#         st.error(
#             recommendation
#         )

#     elif risk_level == "Medium Risk":

#         st.warning(
#             recommendation
#         )

#     else:

#         st.success(
#             recommendation
#         )

#     st.markdown(
#         f"""
#         The **{PREDICTIVE_MODEL_NAME}** estimates that
#         this engine has approximately
#         **{st.session_state.prediction_rul:.2f}
#         operational cycles** remaining.

#         The prediction corresponds to a
#         **{st.session_state.prediction_health_status}**
#         health condition and a
#         **{st.session_state.prediction_risk_level}**
#         risk level.

#         The recommendation translates the numerical
#         model output into operational maintenance
#         decision support. Final maintenance action
#         should remain subject to engineering inspection,
#         operating context and organisational procedures.
#         """
#     )


#     # ========================================================
#     # Explainability reference
#     # ========================================================

#     section_title(
#         "9. Explainability Reference"
#     )

#     xai_col1, xai_col2 = st.columns(2)

#     with xai_col1:

#         st.metric(
#             "Primary XAI Method",
#             PRIMARY_XAI_METHOD,
#         )

#         st.markdown(
#             """
#             TimeSHAP provides sequence-aware
#             feature, event and temporal explanations
#             directly for the Improved LSTM.
#             """
#         )

#     with xai_col2:

#         st.metric(
#             "Complementary XAI Method",
#             SUPPLEMENTARY_XAI_METHOD,
#         )

#         st.markdown(
#             """
#             Integrated Gradients provides an
#             independent gradient-based attribution
#             perspective for comparison with TimeSHAP.
#             """
#         )

#     st.info(
#         """
#         The Explainability module presents validated
#         TimeSHAP and Integrated Gradients artefacts
#         generated during the experimental research phase.

#         For the representative Critical engine case,
#         these explanation artefacts correspond to the
#         same Improved LSTM model and temporal sequence
#         used in the research analysis.
#         """
#     )


# # ============================================================
# # Initial guidance
# # ============================================================

# else:

#     st.divider()

#     st.info(
#         f"""
#         Upload exactly **{WINDOW_SIZE} consecutive
#         operational cycles** for one engine and then
#         select **Predict Remaining Useful Life**.

#         The application will scale the
#         {len(selected_features)} selected variables,
#         construct the LSTM input sequence and generate
#         the RUL prediction, health classification,
#         risk assessment and maintenance recommendation.
#         """
#     )


# end_page()