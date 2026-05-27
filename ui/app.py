import sys
import os
import streamlit as st
from PIL import Image
import tempfile
from auth import register_user, login_user, reset_password

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from inference.predict_image_fixed import predict_image
from inference.predict_video import predict_video

st.set_page_config(
    page_title="🛡️ DeepShield",
    page_icon="🛡️",
    layout="centered"
)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "page" not in st.session_state:
    st.session_state.page = "login"

def _safe_remove_temp_file(path):
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def _run_prediction(predict_fn, media_path, media_label):
    try:
        return predict_fn(media_path)
    except FileNotFoundError as exc:
        st.error(f"{media_label} model file is missing: {exc}")
    except ValueError as exc:
        st.error(f"Unable to process the uploaded {media_label.lower()}: {exc}")
    except Exception as exc:
        st.error(f"Something went wrong while analyzing the {media_label.lower()}: {exc}")
    return None

def login_page():
    st.title("🔐 Login")

    username = st.text_input("👤 Username")
    password = st.text_input("🔑 Password", type="password")

    if st.button("🔓 Login"):
        if login_user(username, password):
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("❌ Invalid credentials")

    if st.button("🔁 Forgot Password?"):
        st.session_state.page = "forgot"
        st.rerun()

    if st.button("📝 Register"):
        st.session_state.page = "register"
        st.rerun()


def register_page():
    st.title("📝 Create Account")

    username = st.text_input("👤 New Username")
    password = st.text_input("🔑 New Password", type="password")
    confirm = st.text_input("🔒 Confirm Password", type="password")

    if st.button("✅ Register"):
        if password != confirm:
            st.error("❌ Passwords do not match")
        else:
            success, msg = register_user(username, password)
            if success:
                st.success(f"✅ {msg}")
                st.session_state.page = "login"
                st.rerun()
            else:
                st.error(f"❌ {msg}")

    if st.button("⬅️ Back to Login"):
        st.session_state.page = "login"
        st.rerun()

def main_app():
    # ---------------- DARK THEME CSS ----------------
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: white;
    }
    .block-container {
        padding-top: 1.5rem;
    }
    h1, h2, h3 {
        text-align: center;
    }
    .result-box {
        padding: 20px;
        border-radius: 12px;
        background: #1c1f26;
        margin-top: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1>🛡️ DeepShield</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center;'>🤖 AI-Powered Deepfake Detection for Images & Videos 🎥🖼️</p>",
        unsafe_allow_html=True
    )

    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.page = "login"
        st.rerun()

    st.divider()

    st.sidebar.title("⚙️ Settings")
    option = st.sidebar.radio("🎞️ Select Media Type", ["Image", "Video"])
    st.sidebar.markdown("---")
    st.sidebar.info("💡 Upload clear face images or videos for best accuracy.")

    if option == "Image":
        st.subheader("🖼️ Image Deepfake Detection")

        uploaded = st.file_uploader(
            "📤 Upload an image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded is not None:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

            img_path = None
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                image.save(tmp.name)
                img_path = tmp.name

            try:
                with st.spinner("🔍 Analyzing image..."):
                    result = _run_prediction(predict_image, img_path, "Image")
            finally:
                _safe_remove_temp_file(img_path)

            display_result(result, media_type="image")

    else:
        st.subheader("🎥 Video Deepfake Detection")

        uploaded = st.file_uploader(
            "📤 Upload a video",
            type=["mp4"]
        )

        if uploaded is not None:
            st.video(uploaded)

            video_path = None
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded.getbuffer())
                video_path = tmp.name

            try:
                with st.spinner("🔍 Analyzing video..."):
                    result = _run_prediction(predict_video, video_path, "Video")
            finally:
                _safe_remove_temp_file(video_path)

            display_result(result, media_type="video")


def display_result(result, media_type):
    if result is None:
        return

    if result["label"] == "NO FACE DETECTED":
        st.error("🚫 No face detected. Please upload a clear face image or video.")
        return

    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
    st.subheader("📊 Prediction Result")

    st.metric("Prediction", f"🧠 {result['label']}")
    st.progress(result["fake_prob"] / 100)

    st.write(f"**✅ Real:** {result['real_prob']}%")
    st.write(f"**❌ Fake:** {result['fake_prob']}%")
    st.write(f"**📏 Confidence Gap:** {result.get('confidence_gap', abs(result['real_prob'] - result['fake_prob']))}%")

    if media_type == "video":
        st.write(f"**🎞️ Face Frames Used:** {result.get('face_frames_used', 0)}")
        st.write(f"**🎬 Sampled Frames:** {result.get('sampled_frames', 0)}")

    if result["label"] == "FAKE":
        st.error("⚠️ This media is likely a deepfake.")
    elif result["label"] == "UNCERTAIN":
        threshold = result.get("confidence_threshold", 5.0)
        reason = result.get("decision_reason", "The model does not have enough evidence for a confident decision.")
        st.warning(
            f"🤔 {reason} "
            f"Gap: {result.get('confidence_gap', 0)}% | Threshold: {threshold}%"
        )
    else:
        st.success("✅ This media appears real.")

    st.markdown("</div>", unsafe_allow_html=True)


def forgot_password_page():
    st.title("🔁 Reset Password")

    username = st.text_input("👤 Enter your username")
    new_pass = st.text_input("🔑 New password", type="password")
    confirm_pass = st.text_input("🔒 Confirm password", type="password")

    if st.button("✅ Reset Password"):
        if new_pass != confirm_pass:
            st.error("❌ Passwords do not match")
        else:
            success, msg = reset_password(username, new_pass)
            if success:
                st.success(f"✅ {msg}")
                st.session_state.page = "login"
                st.rerun()
            else:
                st.error(f"❌ {msg}")

    if st.button("⬅️ Back to Login"):
        st.session_state.page = "login"
        st.rerun()


if not st.session_state.logged_in:

    if st.session_state.page == "login":
        login_page()

    elif st.session_state.page == "register":
        register_page()

    elif st.session_state.page == "forgot":
        forgot_password_page()

else:
    main_app()
