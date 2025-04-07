import streamlit as st
from transformers import pipeline
import random
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# --- Intent Model Setup ---
intent_data = {
    "text": [
        "I feel stressed", "I'm anxious", "Hello", "Hi",
        "I'm sad", "Feeling low", "I'm overwhelmed", "Good morning"
    ],
    "intent": [
        "stress", "anxiety", "greeting", "greeting",
        "sadness", "sadness", "stress", "greeting"
    ]
}
df = pd.DataFrame(intent_data)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
clf = LogisticRegression()
clf.fit(X, df['intent'])

responses = {
    "greeting": ["Hi there! How can I support you today?"],
    "stress": ["Stress is tough. Let’s take a deep breath together."],
    "anxiety": ["It’s okay to feel anxious. I'm here with you."],
    "sadness": ["I’m sorry you feel this way. Want to talk more about it?"],
    "default": ["Tell me more, I'm listening."]
}

recommendations = {
    "stress": [
        {"title": "5-Minute Meditation", "url": "https://www.youtube.com/watch?v=inpok4MKVLM"},
        {"title": "Stretch & Relax", "url": "https://www.youtube.com/watch?v=qHJ992N-Dhs"}
    ],
    "anxiety": [
        {"title": "Anxiety Relief Music", "url": "https://www.youtube.com/watch?v=ZToicYcHIOU"},
        {"title": "Grounding Exercise", "url": "https://www.youtube.com/watch?v=KZXT7L4s0bY"}
    ],
    "sadness": [
        {"title": "Uplifting Music", "url": "https://www.youtube.com/watch?v=UfcAVejslrU"},
        {"title": "Talk on Depression", "url": "https://www.youtube.com/watch?v=XiCrniLQGYc"}
    ],
    "default": [
        {"title": "Mental Health Playlist", "url": "https://www.youtube.com/playlist?list=PLFzWFredxyJlR9L1_LPODw_JH6XkUnYVX"}
    ]
}

sentiment_analyzer = pipeline("sentiment-analysis")
crisis_keywords = ["suicide", "kill myself", "end it", "hopeless", "give up"]
log_data = []

# Function to process input
def process_input(text):
    sentiment = sentiment_analyzer(text)[0]
    if any(word in text.lower() for word in crisis_keywords):
        return "[\u26a0\ufe0f Crisis Detected] Please seek immediate help.", "Crisis"

    input_vec = vectorizer.transform([text])
    intent = clf.predict(input_vec)[0] if clf.predict_proba(input_vec).max() > 0.4 else "default"
    response = random.choice(responses[intent])
    log_data.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "user_input": text,
        "sentiment": sentiment['label'],
        "sentiment_score": sentiment['score'],
        "intent": intent,
        "response": response
    })
    return f"[{sentiment['label']}] {response}", intent

# --- Streamlit Interface ---
st.set_page_config(page_title="Mental Health Assistant", layout="centered")
st.title("🧠 Personalized Mental Health Assistant")

user_input = st.text_input("How are you feeling today?")
if user_input:
    output, intent = process_input(user_input)
    st.markdown(f"**Bot:** {output}")

    st.subheader("🎯 Personalized Recommendations")
    for rec in recommendations.get(intent, recommendations["default"]):
        st.markdown(f"- [{rec['title']}]({rec['url']})")

if st.checkbox("📜 Show Chat History"):
    if log_data:
        st.dataframe(pd.DataFrame(log_data))
    else:
        st.info("No conversation history yet.")

if st.checkbox("📈 Show Mood Timeline"):
    if log_data:
        df_log = pd.DataFrame(log_data)
        df_log["sentiment_num"] = df_log["sentiment"].map({
            "POSITIVE": 1,
            "NEGATIVE": -1,
            "NEUTRAL": 0
        })
        st.line_chart(df_log.set_index("time")["sentiment_num"])
    else:
        st.info("Start chatting to see your mood timeline.")

st.header("📊 Survey Data Analysis")
survey_file = st.file_uploader("Upload your Google Form survey CSV", type="csv")

if survey_file is not None:
    survey_df = pd.read_csv(survey_file)
    st.write("📝 Survey Preview", survey_df.head())

    if "Rate your overall mood (1–5)" in survey_df.columns:
        avg_mood = survey_df["Rate your overall mood (1–5)"].mean()
        st.metric("🌤️ Average Mood Score", round(avg_mood, 2))

        st.subheader("Mood Score Distribution")
        fig, ax = plt.subplots()
        sns.histplot(survey_df["Rate your overall mood (1–5)"], bins=5, ax=ax)
        st.pyplot(fig)

    if "What are you struggling with lately?" in survey_df.columns:
        text = " ".join(survey_df["What are you struggling with lately?"].dropna())
        wc = WordCloud(width=800, height=400, background_color='white').generate(text)
        st.subheader("🧠 Common Concerns")
        st.image(wc.to_array())
