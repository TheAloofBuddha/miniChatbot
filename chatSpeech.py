import streamlit as st
import nltk
nltk.data.path.append(r"C:\Users\PC\AppData\Roaming\nltk_data")
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.metrics import jaccard_distance
import string
from sentence_transformers import SentenceTransformer, util
import speech_recognition as sr

with open(r'C:\Users\PC\Documents\data science python\chatbot\The Project Gutenberg eBook of Moby.txt', 'r', encoding='utf-8') as f:
    data = f.read().replace('\n', ' ')
    
model = SentenceTransformer('all-MiniLM-L6-v2')
sentences = sent_tokenize(data)
stopwords = set(stopwords.words('english'))

corpus = model.encode(sentences, convert_to_tensor=True)

def get_most_relevant_sentence(query, corpus, sentences):
  query_embedded = model.encode(query, convert_to_tensor=True)
  similarities = util.cos_sim(query_embedded, corpus)
  top_index = similarities.argmax().item()
  most_relevant_sentence = sentences[top_index]
  return most_relevant_sentence

def chatbot(question):
    if question:
        response = get_most_relevant_sentence(question, corpus, sentences)
        if response:
            return response
        else:
            return "I'm sorry, I don't have an answer for that."
    else:
        return "Please ask a question."

def transcribe_speech(language="en-US"):
    r = sr.Recognizer()
    r.pause_threshold = 2
    with sr.Microphone() as source:
        st.info("Please speak now...")
        audio = r.listen(source, timeout=10)
    try:
        text = r.recognize_google(audio, language=language)
        return text
    except sr.UnknownValueError:
        return "Sorry, I could not understand the audio."
    except sr.RequestError as e:
        return f"Could not request results from speech service; {e}"

def main():
    st.title("The Answerer")
    st.write("The Answerer is here to provide any information you need from the Moby Dick andf the Whale book")

    options = st.selectbox("Select your input option: ", ["Speech Input", "Text Input"])

    if options == "Speech Input":
        if st.button("Start Speaking"): 
            spoken_text = transcribe_speech()
            st.write("You said:", spoken_text)
            if not spoken_text.startswith("Sorry") and "request" not in spoken_text:
                response = chatbot(spoken_text)
                st.write("🧠 The Answerer: " + response)

    elif options == "Text Input":
        question = st.text_input("Reader: ")
        if st.button("Submit"):
            response = chatbot(question)
            st.write("The Answerer: " + response)

if __name__ == "__main__":
    main()