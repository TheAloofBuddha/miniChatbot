import sys
sys.modules['torch.classes'] = None
import streamlit as st
import nltk
nltk.data.path.append(r"C:\Users\Psalm\AppData\Roaming\nltk_data")
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.metrics import jaccard_distance
import string
from sentence_transformers import SentenceTransformer, util

with open(r'C:\Users\Psalm\Downloads\chatbot\moby_dick.txt', 'r', encoding='utf-8') as f:
    data = f.read().replace('\n', ' ')
    
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
sentences = sent_tokenize(data)
stopwords = set(stopwords.words('english'))

# def preprocessing(sentence):
#   words = word_tokenize(sentence)
#   words = [word.lower() for word in words if word.isalnum()]
#   words = [word for word in words if word not in stopwords]
#   lemmatizer = WordNetLemmatizer()
#   words = [lemmatizer.lemmatize(word) for word in words]
#   return words

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

def main():
    st.title("The Answerer")
    st.write("The Answerer is here to provide any information you need from the Moby Dick and the Whale book")

    question = st.text_input("Reader: ")
    if st.button("Submit"):
        with st.spinner("Thinking..."):
          response = chatbot(question)
          st.write("The Answerer: " + response)
if __name__ == "__main__":
    main()