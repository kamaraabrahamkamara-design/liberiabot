import streamlit as st
from google import genai
from google.genai import types
from PIL import Image

# 1. Page Configuration
st.set_page_config(page_title="Liberia AI Chatbot", page_icon="🇱🇷", layout="wide")
st.title("🇱🇷 Liberia AI Chatbot")

# 2. Securely Initialize the Gemini Client
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=api_key)
except KeyError:
    st.error("Missing API Key! Please add 'GEMINI_API_KEY' to your Streamlit secrets.")
    st.stop()

# 3. Initialize Conversation History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. Multimodal Inputs Sidebar
with st.sidebar:
    st.header("📸 Media Inputs")
    st.write("Capture or upload media to include in your next prompt.")
    
    # Photo Capture Widget
    captured_photo = st.camera_input("Take a photo")
    
    # Voice Recorder Widget
    recorded_voice = st.audio_input("Record a voice message")
    
    # Video Upload Widget
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "webm"])

# 5. Render Chat History on App Rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Re-render media attachments if they were saved in history
        if "media_previews" in message:
            for media_type, data in message["media_previews"].items():
                if media_type == "image":
                    st.image(data, caption="Attached Image", width=300)
                elif media_type == "audio":
                    st.audio(data, format="audio/wav")
                elif media_type == "video":
                    st.video(data)

# 6. Handle New User Input
if user_input := st.chat_input("Ask me anything about your prompt or media..."):
    # Render user text
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Prepare message payload tracking
    current_user_message = {"role": "user", "content": user_input, "media_previews": {}}
    
    # Setup our Gemini prompt materials list
    gemini_payload = []
    
    # Handle Photo Input
    if captured_photo:
        pil_image = Image.open(captured_photo)
        gemini_payload.append(pil_image)
        current_user_message["media_previews"]["image"] = pil_image
        with st.chat_message("user"):
            st.image(pil_image, caption="Attached Image Preview", width=300)

    # Handle Audio Input
    if recorded_voice:
        audio_bytes = recorded_voice.read()
        # Pass audio bytes directly to the client library
        gemini_payload.append(
            types.Part.from_bytes(
                data=audio_bytes,
                mime_type="audio/wav"
            )
        )
        current_user_message["media_previews"]["audio"] = audio_bytes
        with st.chat_message("user"):
            st.audio(audio_bytes, format="audio/wav")

    # Handle Video Input
    if uploaded_video:
        video_bytes = uploaded_video.read()
        # Guess mime type safely (defaulting to video/mp4)
        mime_type = uploaded_video.type if uploaded_video.type else "video/mp4"
        gemini_payload.append(
            types.Part.from_bytes(
                data=video_bytes,
                mime_type=mime_type
            )
        )
        current_user_message["media_previews"]["video"] = video_bytes
        with st.chat_message("user"):
            st.video(video_bytes)

    # Append text prompt at the end of payloads
    gemini_payload.append(user_input)
    st.session_state.messages.append(current_user_message)

    # Process and Stream the response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # We pass the media array directly inside the contents payload
            response_stream = client.models.generate_content_stream(
                model='gemini-3.8-flash',
                contents=gemini_payload,
                config=types.GenerateContentConfig(
                    system_instruction="You are a helpful assistant. You can see images/videos and listen to audio clips provided by the user."
                )
            )
            
            for chunk in response_stream:
                if chunk.text:
                    full_response += chunk.text
                    message_placeholder.markdown(full_response + "▌")
                
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"An error occurred: {e}")