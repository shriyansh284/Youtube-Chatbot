import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
# 1️⃣ Load API Key
# ------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY not found! Please add it to your .env file.")
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ------------------------------
# 2️⃣ Streamlit Page Config
# ------------------------------
st.set_page_config(page_title="🎬 YouTube Q&A Chat", page_icon="🎥")

st.title("🎬 YouTube Video Q&A (Hugging Face + OpenAI)")
st.write(
    "Fetch a YouTube transcript and ask questions — "
    "Hugging Face for embeddings 💪, OpenAI GPT for answers 🧠."
)

# ------------------------------
# 3️⃣ Input: YouTube Video ID
# ------------------------------
video_id = st.text_input("Enter YouTube Video ID:", placeholder="e.g. LPZh9BOjkQs")

if video_id:
    try:
        api = YouTubeTranscriptApi()
        transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
        transcript = " ".join([getattr(t, "text", "") for t in transcript_list])

        st.success("✅ Transcript fetched successfully!")
    except Exception as e:
        st.error(f"❌ Error fetching transcript: {e}")
        st.stop()

    # ------------------------------
    # 4️⃣ Split Transcript into Chunks
    # ------------------------------
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.create_documents([transcript])

    # ------------------------------
    # 5️⃣ Create Hugging Face Embeddings
    # ------------------------------
    st.info("🧩 Creating embeddings using Hugging Face MiniLM model...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embedding_model)
    retriever = vector_store.as_retriever()

    # ------------------------------
    # 6️⃣ Load OpenAI GPT Model
    # ------------------------------
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    # ------------------------------
    # 7️⃣ Ask a Question
    # ------------------------------
    query = st.text_input("💬 Ask a question about this video:")

    if query:
        with st.spinner("🤔 Thinking..."):
            result = qa_chain.invoke({"query": query})
            st.markdown("### 🧠 Answer:")
            st.write(result["result"])

            with st.expander("📜 Relevant transcript sections"):
                for doc in result["source_documents"]:
                    st.write(doc.page_content)

