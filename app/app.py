# app entry point placeholder
import sys
from pathlib import Path
import tempfile

import streamlit as st
import torch
from PIL import Image


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


# ============================================================
# PROJECT MODULES
# ============================================================

from src.detection.detector import (
    RoadObjectDetector
)

from src.scene.scene_graph import (
    SceneGraphBuilder
)

from src.risk.risk_analyzer import (
    RiskAnalyzer
)

from src.nlp.vlm import (
    RoadSceneVLM
)

from src.nlp.report_generator import (
    RoadSceneReportGenerator
)

from src.visualization.visualizer import (
    RoadSceneVisualizer
)


# ============================================================
# UI COMPONENTS
# ============================================================

from components import (
    render_header,
    render_section_title,
    render_metric_card,
    render_risk_card,
    render_object_summary,
    render_relationship_summary,
    render_risk_events,
    render_vlm_description,
    render_report,
    render_pipeline,
    render_footer
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Road Scene Understanding",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# LOAD CSS
# ============================================================

CSS_PATH = (
    Path(__file__).parent
    / "styles.css"
)

if CSS_PATH.exists():

    with open(
        CSS_PATH,
        "r",
        encoding="utf-8"
    ) as css_file:

        st.markdown(
            f"""
            <style>
            {css_file.read()}
            </style>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = (
    r"C:\Users\Suprakash Ghosh\runs\detect\models"
    r"\yolov11\bdd100k_final-3\weights\best.pt"
)

DEFAULT_CONFIDENCE = 0.60
DEFAULT_MAX_DETECTIONS = 80


# ============================================================
# CACHE MODELS
# ============================================================

@st.cache_resource
def load_detector():

    return RoadObjectDetector(
        MODEL_PATH
    )


@st.cache_resource
def load_vlm():

    return RoadSceneVLM()


# ============================================================
# SCENE GRAPH HELPER
# ============================================================

def extract_scene_graph_data(
    scene_graph
):

    if isinstance(
        scene_graph,
        dict
    ):

        nodes = scene_graph.get(
            "nodes",
            []
        )

        edges = scene_graph.get(
            "edges",
            []
        )

    else:

        nodes = getattr(
            scene_graph,
            "nodes",
            []
        )

        edges = getattr(
            scene_graph,
            "edges",
            []
        )

    return nodes, edges


# ============================================================
# HEADER
# ============================================================

render_header()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ System Settings"
    )

    st.subheader(
        "Detection"
    )

    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.90,
        value=DEFAULT_CONFIDENCE,
        step=0.05
    )

    max_detections = st.number_input(
        "Maximum Detections",
        min_value=1,
        max_value=100,
        value=DEFAULT_MAX_DETECTIONS,
        step=1
    )

    st.divider()

    st.subheader(
        "🖥️ Hardware"
    )

    if torch.cuda.is_available():

        st.success(
            "CUDA Available"
        )

        st.caption(
            torch.cuda.get_device_name(0)
        )

        st.caption(
            f"CUDA {torch.version.cuda}"
        )

    else:

        st.warning(
            "CUDA unavailable"
        )

        st.caption(
            "The system will use CPU."
        )

    st.divider()

    st.subheader(
        "🧠 AI Pipeline"
    )

    st.caption(
        "🎯 YOLO11 Detection"
    )

    st.caption(
        "🧠 Scene Graph"
    )

    st.caption(
        "⚠️ Risk Analysis"
    )

    st.caption(
        "🤖 Vision-Language Model"
    )

    st.caption(
        "📄 AI Report"
    )


# ============================================================
# UPLOAD
# ============================================================

render_section_title(
    "📷 Upload Road Scene"
)

st.markdown(
    """
    <div class="upload-description">
        Upload a JPG, JPEG, or PNG road image
        for complete AI analysis.
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose a road image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# NO IMAGE
# ============================================================

if uploaded_file is None:

    render_pipeline()

    render_footer()

    st.stop()


# ============================================================
# DISPLAY INPUT
# ============================================================

image = Image.open(
    uploaded_file
).convert("RGB")

col1, col2 = st.columns(
    [2, 1]
)

with col1:

    st.image(
        image,
        caption="Input Road Scene",
        use_container_width=True
    )

with col2:

    st.markdown(
        """
        <div class="info-card">
            <div class="info-card-title">
                📋 Image Information
            </div>
        """,
        unsafe_allow_html=True
    )

    st.write(
        f"**File:** {uploaded_file.name}"
    )

    st.write(
        f"**Width:** {image.width}px"
    )

    st.write(
        f"**Height:** {image.height}px"
    )

    st.write(
        f"**Confidence:** {confidence:.2f}"
    )

    st.write(
        f"**Max detections:** {max_detections}"
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================
# ANALYZE BUTTON
# ============================================================

st.divider()

analyze = st.button(
    "🚀 Analyze Road Scene",
    type="primary",
    use_container_width=True
)


# ============================================================
# ANALYSIS
# ============================================================

if analyze:

    temporary_path = None

    try:

        # ====================================================
        # TEMPORARY IMAGE
        # ====================================================

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg"
        ) as temp_file:

            image.save(
                temp_file.name,
                format="JPEG"
            )

            temporary_path = Path(
                temp_file.name
            )


        # ====================================================
        # MODEL LOADING
        # ====================================================

        with st.spinner(
            "Loading AI models..."
        ):

            detector = load_detector()

            vlm = load_vlm()


        # ====================================================
        # PROGRESS
        # ====================================================

        progress = st.progress(
            0
        )

        status = st.empty()


        # ====================================================
        # STEP 1 — YOLO11
        # ====================================================

        status.info(
            "🔍 Step 1/5 — YOLO11 object detection..."
        )

        progress.progress(
            15
        )

        results = detector.detect(
            str(temporary_path),
            confidence=confidence
        )

        result = results[0]

        detections = []

        boxes = result.boxes

        for i in range(
            len(boxes)
        ):

            cls_id = int(
                boxes.cls[i].item()
            )

            confidence_value = float(
                boxes.conf[i].item()
            )

            x1, y1, x2, y2 = (
                boxes.xyxy[i].tolist()
            )

            class_name = (
                detector.model.names[
                    cls_id
                ]
            )

            detections.append(
                {
                    "id": i + 1,
                    "class_name": class_name,
                    "confidence": confidence_value,
                    "bbox": [
                        x1,
                        y1,
                        x2,
                        y2
                    ]
                }
            )


        # Sort detections

        detections = sorted(
            detections,
            key=lambda x: x["confidence"],
            reverse=True
        )

        detections = detections[
            :int(max_detections)
        ]


        # ====================================================
        # STEP 2 — SCENE GRAPH
        # ====================================================

        status.info(
            "🧠 Step 2/5 — Building scene graph..."
        )

        progress.progress(
            35
        )

        image_width = (
            result.orig_shape[1]
        )

        image_height = (
            result.orig_shape[0]
        )

        scene_builder = (
            SceneGraphBuilder()
        )

        scene_graph = (
            scene_builder.build(
                detections=detections,
                image_width=image_width,
                image_height=image_height
            )
        )

        nodes, edges = (
            extract_scene_graph_data(
                scene_graph
            )
        )


        # ====================================================
        # STEP 3 — RISK
        # ====================================================

        status.info(
            "⚠️ Step 3/5 — Running risk analysis..."
        )

        progress.progress(
            55
        )

        risk_analyzer = (
            RiskAnalyzer()
        )

        risk_result = (
            risk_analyzer.analyze(
                scene_graph
            )
        )

        risk_level = risk_result.get(
            "risk_level",
            "UNKNOWN"
        )

        risk_score = risk_result.get(
            "risk_score",
            0
        )


        # ====================================================
        # STEP 4 — VLM
        # ====================================================

        status.info(
            "🤖 Step 4/5 — Generating VLM description..."
        )

        progress.progress(
            75
        )

        caption = (
            vlm.generate_caption(
                temporary_path
            )
        )


        # ====================================================
        # STEP 5 — REPORT
        # ====================================================

        status.info(
            "📄 Step 5/5 — Generating AI report..."
        )

        progress.progress(
            90
        )

        report_generator = (
            RoadSceneReportGenerator()
        )

        report = (
            report_generator.generate_report(
                caption=caption,
                detections=detections,
                scene_graph=scene_graph,
                risk_result=risk_result
            )
        )


        # ====================================================
        # VISUALIZATION
        # ====================================================

        output_dir = (
            PROJECT_ROOT
            / "outputs"
            / "visualizations"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_path = (
            output_dir
            / "streamlit_analysis.jpg"
        )

        visualizer = (
            RoadSceneVisualizer()
        )

        visualizer.visualize(
            image_path=temporary_path,
            detections=detections,
            risk_result=risk_result,
            output_path=output_path,
            scene_graph=scene_graph
        )


        progress.progress(
            100
        )

        status.success(
            "✅ Complete analysis finished!"
        )


        # ====================================================
        # RESULTS
        # ====================================================

        st.divider()

        render_section_title(
            "📊 Analysis Results"
        )


        # ====================================================
        # METRICS
        # ====================================================

        metric1, metric2, metric3, metric4 = (
            st.columns(4)
        )

        with metric1:

            render_metric_card(
                "OBJECTS",
                len(detections)
            )

        with metric2:

            render_metric_card(
                "RELATIONSHIPS",
                len(edges)
            )

        with metric3:

            render_metric_card(
                "RISK SCORE",
                f"{risk_score}/100"
            )

        with metric4:

            render_risk_card(
                risk_level
            )


        # ====================================================
        # VISUALIZATION
        # ====================================================

        st.divider()

        render_section_title(
            "🎯 AI Visualization"
        )

        st.image(
            str(output_path),
            caption=(
                "YOLO11 + Scene Graph + Risk Analysis"
            ),
            use_container_width=True
        )


        # ====================================================
        # OBJECTS + RELATIONSHIPS
        # ====================================================

        st.divider()

        left_col, right_col = (
            st.columns(2)
        )

        with left_col:

            render_object_summary(
                detections
            )

        with right_col:

            render_relationship_summary(
                edges
            )


        # ====================================================
        # VLM
        # ====================================================

        st.divider()

        render_vlm_description(
            caption
        )


        # ====================================================
        # RISK EVENTS
        # ====================================================

        st.divider()

        render_risk_events(
            risk_result
        )


        # ====================================================
        # REPORT
        # ====================================================

        st.divider()

        render_report(
            report
        )


        # ====================================================
        # DOWNLOAD IMAGE
        # ====================================================

        with open(
            output_path,
            "rb"
        ) as image_file:

            st.download_button(
                label="🖼️ Download Visualization",
                data=image_file,
                file_name=(
                    "road_scene_visualization.jpg"
                ),
                mime="image/jpeg",
                use_container_width=True
            )


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as error:

        st.error(
            "❌ Analysis failed."
        )

        st.exception(
            error
        )


    # ========================================================
    # CLEANUP
    # ========================================================

    finally:

        if (
            temporary_path is not None
            and temporary_path.exists()
        ):

            try:

                temporary_path.unlink()

            except Exception:

                pass


# ============================================================
# FOOTER
# ============================================================

render_footer()