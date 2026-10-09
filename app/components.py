# components placeholder
import streamlit as st


# ============================================================
# HEADER
# ============================================================

def render_header():

    st.markdown(
        """
        <div class="main-title">
            🚗 AI Road Scene Understanding
        </div>

        <div class="subtitle">
            YOLO11 • Scene Graph • Risk Analysis • Vision-Language Model
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# SECTION TITLE
# ============================================================

def render_section_title(
    title
):

    st.markdown(
        f"""
        <div class="section-title">
            {title}
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# METRIC CARD
# ============================================================

def render_metric_card(
    label,
    value,
    css_class=""
):

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                {label}
            </div>

            <div class="metric-value {css_class}">
                {value}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# RISK CARD
# ============================================================

def render_risk_card(
    risk_level
):

    level = str(
        risk_level
    ).upper()

    if level == "HIGH":

        css_class = "risk-high"

    elif level == "MEDIUM":

        css_class = "risk-medium"

    elif level == "LOW":

        css_class = "risk-low"

    else:

        css_class = "risk-safe"

    render_metric_card(
        "RISK LEVEL",
        level,
        css_class
    )


# ============================================================
# OBJECT SUMMARY
# ============================================================

def render_object_summary(
    detections
):

    st.subheader(
        "🔍 Detected Objects"
    )

    object_counts = {}

    for detection in detections:

        class_name = detection[
            "class_name"
        ]

        object_counts[class_name] = (
            object_counts.get(
                class_name,
                0
            ) + 1
        )

    if not object_counts:

        st.info(
            "No objects detected."
        )

        return

    for class_name, count in (
        object_counts.items()
    ):

        st.write(
            f"**{class_name.title()}** — {count}"
        )


# ============================================================
# RELATIONSHIP SUMMARY
# ============================================================

def render_relationship_summary(
    edges
):

    st.subheader(
        "🧠 Scene Relationships"
    )

    relationship_counts = {
        "near": 0,
        "overlapping": 0,
        "left_of": 0,
        "above": 0,
        "below": 0
    }

    for edge in edges:

        if isinstance(edge, dict):

            relation = edge.get(
                "relation",
                edge.get(
                    "relationship"
                )
            )

        else:

            relation = getattr(
                edge,
                "relation",
                getattr(
                    edge,
                    "relationship",
                    None
                )
            )

        if relation in relationship_counts:

            relationship_counts[
                relation
            ] += 1

    found = False

    for relation, count in (
        relationship_counts.items()
    ):

        if count > 0:

            found = True

            st.write(
                f"**{relation}** — {count}"
            )

    if not found:

        st.info(
            "No spatial relationships detected."
        )


# ============================================================
# RISK EVENTS
# ============================================================

def render_risk_events(
    risk_result
):

    st.subheader(
        "⚠️ Risk Events"
    )

    events = risk_result.get(
        "events",
        []
    )

    if not events:

        st.success(
            "No significant risk events detected."
        )

        return

    for event in events:

        event_type = event.get(
            "type",
            event.get(
                "event",
                "Unknown Event"
            )
        )

        severity = event.get(
            "severity",
            "UNKNOWN"
        )

        score = event.get(
            "score",
            0
        )

        reason = event.get(
            "reason",
            ""
        )

        with st.expander(
            f"{event_type} | {severity}"
        ):

            st.write(
                f"**Score:** {score}"
            )

            st.write(
                f"**Reason:** {reason}"
            )


# ============================================================
# VLM DESCRIPTION
# ============================================================

def render_vlm_description(
    caption
):

    st.subheader(
        "🤖 VLM Scene Description"
    )

    if caption:

        st.info(
            caption
        )

    else:

        st.warning(
            "The VLM did not produce a description."
        )


# ============================================================
# REPORT
# ============================================================

def render_report(
    report
):

    st.subheader(
        "📄 AI Road Scene Report"
    )

    st.text_area(
        "Generated Report",
        report,
        height=420
    )

    st.download_button(
        label="⬇️ Download Analysis Report",
        data=report,
        file_name="road_scene_analysis.txt",
        mime="text/plain",
        use_container_width=True
    )


# ============================================================
# PIPELINE
# ============================================================

def render_pipeline():

    st.markdown(
        """
        ### 🔬 AI Processing Pipeline

        <div class="pipeline-card">
            📷 Road Image
        </div>

        ↓

        <div class="pipeline-card">
            🎯 YOLO11 Object Detection
        </div>

        ↓

        <div class="pipeline-card">
            🧠 Scene Graph Construction
        </div>

        ↓

        <div class="pipeline-card">
            ⚠️ Risk Analysis
        </div>

        ↓

        <div class="pipeline-card">
            🤖 Vision-Language Model
        </div>

        ↓

        <div class="pipeline-card">
            📄 AI Report
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

def render_footer():

    st.markdown(
        """
        <div class="footer">
            AI-Based Road Scene Understanding and Risk Analysis
            <br>
            YOLO11 • Scene Graph • Vision-Language Model
        </div>
        """,
        unsafe_allow_html=True
    )