import os
import re
import numpy as np
import pandas as pd
import streamlit as st
from bertopic import BERTopic
import matplotlib.pyplot as plt


st.set_page_config(page_title="Movie Topic Explorer (BERTopic)", layout="wide")

ARTIFACTS_DIR = "artifacts"
MODEL_DIR = os.path.join(ARTIFACTS_DIR, "bertopic_model")
MOVIES_PATH = os.path.join(ARTIFACTS_DIR, "movies.parquet")  # or csv
EMB2D_PATH = os.path.join(ARTIFACTS_DIR, "reduced_embeddings.npy")
PLOT_DESCRIPTIONS_PATH=  os.path.join(ARTIFACTS_DIR, "topic_map.html") 


@st.cache_resource(show_spinner=True)
def load_topic_model(model_dir: str) -> BERTopic:
    return BERTopic.load(model_dir)


@st.cache_data(show_spinner=True)
def load_movies(path: str) -> pd.DataFrame:
    # Supports parquet or csv
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

  



    # Basic cleanup
    df["Title"] = df["Title"].astype(str)
    df["Summary"] = df["Summary"].astype(str)
    df["Name"] = df["Name"].astype(str)

    # Parse numeric topic id from prefix before first underscore: "13_something" -> 13
    def parse_topic_id(s: str) -> int:
        m = re.match(r"^\s*(\d+)_", s)
        if not m:
            raise ValueError(
                f"Topic label '{s}' does not start with '<int>_'. "
                "Store a numeric 'topic' column or ensure labels start like '13_...'."
            )
        return int(m.group(1))

    df["topic_id"] = df["Name"].apply(parse_topic_id)


    return df


@st.cache_data(show_spinner=True)
def load_reduced_embeddings(path: str, expected_n: int) -> np.ndarray:
    emb2d = np.load(path)

    return emb2d


def get_topic_words(topic_model: BERTopic, topic_id: int, topn: int = 10) -> pd.DataFrame:
    words = topic_model.get_topic(topic_id)
    if not words:
        return pd.DataFrame(columns=["word", "score"])
    return pd.DataFrame(words[:topn], columns=["word", "score"])






# -------------------- UI --------------------
st.title("Movie Topic Explorer (BERTopic)")

with st.sidebar:
    st.header("Controls")
    show_plot = st.checkbox("Show topic map", value=True)
   
    n_similar = st.slider("Other movies to show (same topic)", 10, 200, 50, step=10)
    topn_words = st.slider("Top words to show", 5,7,10 ,step=1)

    st.divider()
    st.caption("Expected artifacts: bertopic_model/, movies.parquet (or csv), reduced_embeddings_2d.npy")


topic_model = load_topic_model(MODEL_DIR)
df = load_movies(MOVIES_PATH)
emb2d = load_reduced_embeddings(EMB2D_PATH, expected_n=len(df))

# Movie selector
selected_title = st.selectbox("Pick a movie", df["Title"].tolist(), index=0)
sel_idx = df.index[df["Title"] == selected_title][0]
row = df.loc[sel_idx]

topic_id = int(row["topic_id"])
topic_label = row["Name"]

left, right = st.columns([0.45, 0.55], gap="large")

with left:
    st.subheader("Selected movie")
    st.write(f"**Title:** {row['Title']}")
    st.write(f"**Topic label:** {topic_label}")

    st.write("**Description:**")
    st.write(row["Summary"])

    st.divider()
    st.subheader("Topic words (BERTopic)")
    topic_words = get_topic_words(topic_model, topic_id, topn=topn_words)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(topic_words["word"], topic_words["score"])
    ax.set_title(f"Top words for topic {topic_id}")
    ax.invert_yaxis()

    st.pyplot(fig)

with right:
    st.subheader("Other movies in the same topic")
    same_topic = df[df["topic_id"] == topic_id].copy().drop(index=sel_idx, errors="ignore")
    st.caption(f"{len(same_topic)} movies found. Showing top {min(n_similar,len(same_topic))}.")
    st.dataframe(
        same_topic[["Title"]].head(n_similar),
        use_container_width=True,
        height=520
    )

st.divider()

if show_plot:
    st.markdown(
        "<div style='text-align:center; font-size:0.875rem; color: rgba(49, 51, 63, 0.6);'>"
        "farklı topiclerin birbirine uzaklığını grafikte labellara tıklayarak görüntüleyebilirsiniz"
        "</div>",
        unsafe_allow_html=True,
    )
    with open(PLOT_DESCRIPTIONS_PATH, "r") as f:
        html = f.read()

    st.components.v1.html(html, height=750, scrolling=True)
