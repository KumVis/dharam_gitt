import os
import streamlit as st
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

# ------------------------
# 1. Load environment variables
# ------------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# ------------------------
# 2. Initialize models
# ------------------------
st.title("🎥 Chat with Your Video using AI")

# Groq LLM (Llama3)
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"
)

# Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Whisper transcription model
whisper_model = WhisperModel("small", device="cpu", compute_type="int8")

# ------------------------
# 3. Video upload
# ------------------------
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])
if uploaded_file:
    video_path = f"temp_{uploaded_file.name}"
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success("✅ Video uploaded successfully!")

    # ------------------------
    # 4. Transcribe the video
    # ------------------------
    st.info("Transcribing video... please wait ⏳")
    segments, info = whisper_model.transcribe(video_path, beam_size=5)

    transcript_texts, timestamps = [], []
    for seg in segments:
        transcript_texts.append(seg.text.strip())
        timestamps.append((seg.start, seg.end))

    transcript = " ".join(transcript_texts)
    st.text_area("📜 Transcript", transcript, height=200)

    # ------------------------
    # 5. Build embeddings index
    # ------------------------
    embeddings = embedder.encode(transcript_texts)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])  # simple L2 similarity search
    index.add(embeddings)

    st.success("✅ Embeddings created and indexed!")

    # ------------------------
    # 6. Chat interface
    # ------------------------
    st.subheader("💬 Chat with your Video")

    user_question = st.text_input("Ask a question about the video:")
    if user_question:
        # Embed user query
        q_embedding = embedder.encode([user_question]).astype("float32")

        # Retrieve top matching segments
        D, I = index.search(q_embedding, k=3)
        retrieved_segments = [transcript_texts[i] for i in I[0]]

        context = "\n".join(retrieved_segments)

        # Construct prompt for LLM
        system_message = SystemMessage(
            content="You are a helpful assistant that answers questions strictly using the provided video transcript. "
        "If the answer is not explicitly present in the transcript, respond with: "
        "'The transcript does not contain this information.' "
        "Do not use outside knowledge or assumptions."
        )
        human_message = HumanMessage(
            content=f"Transcript context:\n{context}\n\nQuestion: {user_question}"
        )

        response = llm([system_message, human_message])

        st.markdown(f"**Answer:** {response.content}")
