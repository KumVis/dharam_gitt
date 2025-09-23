import os
import json
import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

# ----------------------------
# Utility Functions
# ----------------------------
def transcribe_video(video_path: str):
    """Transcribe video into text segments using faster-whisper."""
    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments, info = model.transcribe(video_path, beam_size=5)
    texts, timestamps = [], []
    for seg in segments:
        texts.append(seg.text.strip())
        timestamps.append((seg.start, seg.end))
    return texts, timestamps

def save_transcript(video_path, texts, timestamps):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    out_file = f"{base_name}_transcript.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"texts": texts, "timestamps": timestamps}, f, ensure_ascii=False, indent=2)

def load_transcript(video_path):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    file = f"{base_name}_transcript.json"
    if os.path.exists(file):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["texts"], data["timestamps"]
    return None, None

def save_faiss_index(index, video_path):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    faiss.write_index(index, f"{base_name}_index.faiss")

def load_faiss_index(video_path):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    file = f"{base_name}_index.faiss"
    if os.path.exists(file):
        return faiss.read_index(file)
    return None

def save_embeddings(video_path, embeddings):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    np.save(f"{base_name}_embeddings.npy", embeddings)

def load_embeddings(video_path):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    file = f"{base_name}_embeddings.npy"
    if os.path.exists(file):
        return np.load(file)
    return None

# ----------------------------
# Streamlit App
# ----------------------------
st.set_page_config(page_title="🎥 Chat with Video", layout="wide")

st.title("🎥 Chat with Your Video")
st.markdown("Upload a video, and ask questions based on its transcript!")

# Sidebar - Settings
st.sidebar.header("⚙️ Settings")
llm_model = st.sidebar.selectbox("LLM Model", ["openai/gpt-oss-20b", "llama-3.1-8b-instant", "gemma2-9b-it"])
embedding_model_name = st.sidebar.selectbox("Embedding Model", ["all-MiniLM-L6-v2"])
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.3)
top_k = st.sidebar.slider("Top-K Segments", 1, 10, 5)

# File uploader
uploaded_video = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])
if uploaded_video:
    video_path = os.path.join("uploaded_" + uploaded_video.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())
    st.video(video_path)

    # Load transcript or process new
    texts, timestamps = load_transcript(video_path)
    if texts is None:
        with st.spinner("Transcribing video..."):
            texts, timestamps = transcribe_video(video_path)
            save_transcript(video_path, texts, timestamps)
        st.success("Transcript created ✅")
    else:
        st.info("Loaded transcript from cache ✅")

    transcript_segments = [seg.strip() for seg in texts if seg.strip()]

    # Embeddings + FAISS
    embedder = SentenceTransformer(embedding_model_name)
    index = load_faiss_index(video_path)
    embeddings = load_embeddings(video_path)

    if index is None or embeddings is None:
        with st.spinner("Creating embeddings and FAISS index..."):
            embeddings = embedder.encode(transcript_segments, convert_to_numpy=True).astype("float32")
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings)
            save_faiss_index(index, video_path)
            save_embeddings(video_path, embeddings)
        st.success("Embeddings & FAISS index saved ✅")
    else:
        st.info("Loaded FAISS index & embeddings from cache ✅")

    # Load LLM
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model, temperature=temperature)

    # Chat interface
    st.subheader("💬 Chat with Transcript")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.text_input("Ask a question about the video:")

    if user_question:
        q_embedding = embedder.encode([user_question], convert_to_numpy=True).astype("float32")
        D, I = index.search(q_embedding, k=top_k)
        retrieved_segments = [transcript_segments[i] for i in I[0]]
        context = "\n".join(retrieved_segments)

        system_message = SystemMessage(
            content="You are an expert assistant helping to answer questions using only the provided transcript excerpts. "
                    "Base your answers strictly on the information in the transcript. "
                    "If the answer cannot be found or inferred directly from the transcript, respond with: "
                    "'The transcript does not contain this information.'"
        )

        human_message = HumanMessage(
            content=f"Transcript context:\n{context}\n\nQuestion: {user_question}"
        )

        with st.spinner("Thinking..."):
            response = llm([system_message, human_message])

        # Save chat history
        st.session_state.chat_history.append(("You", user_question))
        st.session_state.chat_history.append(("Assistant", response.content))

    # Display chat
    for speaker, msg in st.session_state.chat_history:
        if speaker == "You":
            st.markdown(f"**🧑‍💻 {speaker}:** {msg}")
        else:
            st.markdown(f"**🤖 {speaker}:** {msg}")
