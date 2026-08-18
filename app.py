import re
import streamlit as st
from transformers import pipeline
 
st.set_page_config(page_title="Bahrain Bites Analyzer", page_icon="📋")
 
TOPICS = ["Food Quality", "Service", "Price", "Cleanliness", "Atmosphere", "Location", "Waiting Time"]
EMOJI = {"positive": "😊", "negative": "😠", "neutral": "😐"}
 
 
def is_arabic(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text))
 
 
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    return {
        "en": pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest"),
        "ar": pipeline("sentiment-analysis", model="CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"),
        "topic": pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"),
    }
 
 
st.title("📋 Bahrain Bites Analyzer")
st.caption("Paste a review in English or Arabic")
 
text = st.text_area("Review text", placeholder="الأكل زين بس الخدمة شوي بطيئة...", height=100)
 
if st.button("Analyze", type="primary") and text.strip():
    models = load_models()
    lang = "ar" if is_arabic(text) else "en"
 
    sentiment = models[lang](text)[0]
    label = sentiment["label"].lower()
 
    topic = models["topic"](text, TOPICS)
 
    col1, col2 = st.columns(2)
    col1.metric("Sentiment", f"{EMOJI.get(label, '')} {label}", f"{sentiment['score']:.0%} confidence")
    col2.metric("Topic", topic["labels"][0], f"{topic['scores'][0]:.0%} confidence")
 
    with st.expander("Raw output"):
        st.json({
            "sentiment": {"label": label, "score": float(sentiment["score"])},
            "topic": {"label": topic["labels"][0], "score": float(topic["scores"][0])},
        })
 
