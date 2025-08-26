import streamlit as st
import nltk
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
from newspaper import Article
from deep_translator import GoogleTranslator
from langdetect import DetectorFactory
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
import re


# ---------- Website configurations---------------
st.set_page_config(
    page_title="My Custom Website Name",  # This changes the tab title
    page_icon="🌐",                       # Optional: adds an icon to the tab
    layout="centered"                     # Optional: layout settings
)


# ---------- NLTK Setup ----------
DetectorFactory.seed = 0

def _ensure_nltk():
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        try:
            nltk.download('punkt_tab', quiet=True)
        except Exception:
            pass

_ensure_nltk()
sia = SentimentIntensityAnalyzer()

# ---------- Sentiment Analysis ----------
def analyze_sentiment(text):
    translator = GoogleTranslator(source="auto", target="en")
    translated = translator.translate(text)

    if translated != text:
        st.write(f"**Original Text:** {text}")
        st.write(f"**Translated to English:** {translated}")

    blob = TextBlob(translated)
    polarity = blob.sentiment.polarity

    vader_scores = sia.polarity_scores(translated)
    vader_compound = vader_scores['compound']

    if polarity > 0.05 or vader_compound > 0.05:
        sentiment = "Positive"
    elif polarity < -0.05 or vader_compound < -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return polarity, vader_compound, sentiment, translated

# ---------- Plot Sentiment ----------
def plot_sentiment(vader_score):
    labels = ['Negative', 'Neutral', 'Positive']
    values = [max(0, -vader_score), 1-abs(vader_score), max(0, vader_score)]

    fig, ax = plt.subplots()
    ax.bar(labels, values, color=['red', 'gray', 'green'])
    ax.set_title('Sentiment Analysis Results')
    ax.set_xlabel('Sentiments')
    ax.set_ylabel('Score')
    st.pyplot(fig)

# ---------- Summarization ----------
def _naive_summarize(text, sentences_count=3):
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return " ".join(sents[:sentences_count])

def summarize_text(text, sentences_count=3):
    text = (text or "").strip()
    if len(text.split()) < 30:
        return text

    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = TextRankSummarizer()
        summary_sents = summarizer(parser.document, sentences_count)
        summary = " ".join(str(s) for s in summary_sents).strip()
        return summary or _naive_summarize(text, sentences_count)
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            summarizer = TextRankSummarizer()
            summary_sents = summarizer(parser.document, sentences_count)
            summary = " ".join(str(s) for s in summary_sents).strip()
            return summary or _naive_summarize(text, sentences_count)
        except Exception:
            return _naive_summarize(text, sentences_count)

# ---------- Word Cloud ----------
def generate_wordcloud(text):
    stopwords = set(STOPWORDS)
    wc = WordCloud(width=800, height=400, stopwords=stopwords,
                   background_color='white').generate(text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

# ---------- Streamlit UI ----------
st.title("AI-assisted eConsultation Feedback Analyzer")
st.subheader("Sentiment Analysis | Summary | Word Cloud")

option = st.sidebar.radio("Choose Analysis Type:",
                          ["Analyzing Text", "Analyzing News Article", "Batch Comments"])

if option == "Analyzing Text":
    user_input = st.text_area("Enter stakeholder comment or feedback:")
    if st.button("Analyze"):
        if user_input.strip():
            polarity, vader, sentiment, processed = analyze_sentiment(user_input)

            tab1, tab2, tab3 = st.tabs(["Sentiment", "Summary", "Word Cloud"])

            with tab1:
                st.success(f"**Overall Sentiment:** {sentiment}")
                st.info(f"TextBlob Polarity: {polarity:.2f} | VADER Score: {vader:.2f}")
                plot_sentiment(vader)

            with tab2:
                summary = summarize_text(processed)
                st.write("**Generated Summary:**")
                st.write(summary)

            with tab3:
                generate_wordcloud(processed)

        else:
            st.warning("Please enter some text!")

elif option == "Analyzing News Article":
    url = st.text_input("Enter News Article URL:")
    if st.button("Analyze"):
        if url.strip():
            try:
                article = Article(url)
                article.download()
                article.parse()
                article.nlp()
                text = article.text

                st.subheader("Extracted Article Summary:")
                st.write(article.summary)

                polarity, vader, sentiment, processed = analyze_sentiment(text)

                tab1, tab2, tab3 = st.tabs(["Sentiment", "Summary", "Word Cloud"])

                with tab1:
                    st.success(f"**Overall Sentiment:** {sentiment}")
                    st.info(f"TextBlob Polarity: {polarity:.2f} | VADER Score: {vader:.2f}")
                    plot_sentiment(vader)

                with tab2:
                    summary = summarize_text(processed)
                    st.write("**Generated Summary:**")
                    st.write(summary)

                with tab3:
                    generate_wordcloud(processed)
            except Exception as e:
                st.error(f"Error processing article: {e}")
        else:
            st.warning("Please enter a valid URL!")

elif option == "Batch Comments":
    st.write("Upload a `.txt` file containing multiple stakeholder comments (one per line).")
    uploaded = st.file_uploader("Upload File", type=["txt"])

    if uploaded:
        content = uploaded.read().decode("utf-8")
        comments = [c.strip() for c in content.split("\n") if c.strip()]

        all_text = " ".join(comments)
        sentiments = []

        for idx, comment in enumerate(comments, 1):
            _, _, sentiment, _ = analyze_sentiment(comment)
            sentiments.append((idx, comment, sentiment))

        st.subheader("Individual Comment Sentiment Results")
        for idx, comment, sentiment in sentiments:
            st.write(f"**Comment {idx}:** {comment}")
            st.info(f"Sentiment: {sentiment}")

        st.subheader("Overall Analysis")
        summary = summarize_text(all_text, sentences_count=5)
        st.write("**Overall Summary:**")
        st.write(summary)

        st.write("**Word Cloud of Stakeholder Comments:**")
        generate_wordcloud(all_text)
