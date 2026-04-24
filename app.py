import streamlit as st
import time
import os
from indexer import build_index
from search import search

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="🎥 AI Video Search Engine",
    layout="wide",
)

# ------------------ HEADER ------------------
st.markdown("""
    <h1 style='text-align: center;'>🎥 Intelligent Video Search</h1>
    <p style='text-align: center; font-size:18px;'>
    Search videos using natural language like <b>"person walking"</b> or <b>"car on road"</b>
    </p>
""", unsafe_allow_html=True)

st.divider()

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.header("⚙️ Controls")

    st.markdown("### 📤 Upload Video")
    video = st.file_uploader("Upload .mp4 file", type=["mp4"])

    if video:
        with open("temp.mp4", "wb") as f:
            f.write(video.read())
        st.success("✅ Video uploaded!")

    st.markdown("### 🚀 Indexing")

    if st.button("Build Index"):
        if not video:
            st.warning("⚠️ Please upload a video first!")
        else:
            with st.spinner("⏳ Processing video... This may take time"):
                build_index("temp.mp4")
            st.success("✅ Index built successfully!")

# ------------------ MAIN SEARCH ------------------
st.markdown("## 🔍 Search")

query = st.text_input("Enter your query (e.g., person, car, people talking)")

col1, col2 = st.columns([1, 1])

with col1:
    top_k = st.slider("Top Results", 1, 10, 5)

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    search_btn = st.button("🔎 Search")

# ------------------ SEARCH LOGIC ------------------
if search_btn:

    if not query:
        st.warning("⚠️ Please enter a query!")
    else:
        start = time.time()

        results = search(query, top_k=top_k)

        end = time.time()

        if not results:
            st.error("❌ No index found. Please upload & index a video first!")
        else:
            st.success(f"⚡ Results found in {round((end-start)*1000, 2)} ms")

            st.divider()
            st.subheader("🎯 Results")

            cols = st.columns(2)

            for i, res in enumerate(results):
                with cols[i % 2]:

                    st.image(res["frame"], use_container_width=True)

                    st.markdown(f"""
                    **⏱ Timestamp:** {round(res['timestamp'], 2)} sec  
                    **🎯 Score:** {round(res['score'], 4)}
                    """)

                    # 🔥 PLAY VIDEO AT THAT TIMESTAMP
                    if os.path.exists("temp.mp4"):
                        st.video("temp.mp4", start_time=int(res["timestamp"]))

# ------------------ FOOTER ------------------
st.divider()
st.markdown("""
<p style='text-align:center; font-size:14px;'>
Built with ❤️ using CLIP + FAISS | Intelligent Video Search Engine
</p>
""", unsafe_allow_html=True)