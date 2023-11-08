import secretkeys

import os
from PyPDF2 import PdfReader
import streamlit as st 

# Imports perform the following (non-exhaustive):
# Measures the relatedness of text strings 
# Split content in pdf 
# Store embedding in vectorstores

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.text_splitter import CharacterTextSplitter

# TODO: Find out how to use this chain to keep track of history
# from langchain.chains import ConversationalRetrievalChain

from langchain.chat_models import ChatOpenAI

from langchain.vectorstores import faiss

from langchain.chains.question_answering import load_qa_chain

def get_llm_response(query, local_chat_history, docs, qa_chain):
    res = qa_chain({"question": query, "chat_history": local_chat_history, "input_documents": docs})
    bot_output = res['output_text']
    print("EChObot:", bot_output)
    local_chat_history.append({query, bot_output})
    return bot_output
    
# Main

#### LANGCHAIN ####
if os.environ.get("OPENAI_API_KEY") == None: 
    os.environ["OPENAI_API_KEY"] = secretkeys.OPENAI_API_KEY

if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello World! AMA about VCS Methodology!"}]


data = PdfReader("VM0007-REDDMF_v1.6.pdf")
combined_text = ''

for page in data.pages:
    combined_text += page.extract_text()

char_text_splitter = CharacterTextSplitter(
    separator='\n',
    chunk_size=1000,
    chunk_overlap=0,
    length_function=len
)

split_text = char_text_splitter.split_text(combined_text)

embeddings = OpenAIEmbeddings()
document_embeddings = faiss.FAISS.from_texts(split_text, embeddings)

local_chat_history = []
query = ""

#### STREAMLIT ####
st.set_page_config(page_title="EChObot")
st.header("EChObot")
st.balloons()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if query := st.chat_input(disabled=False):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

if st.session_state.messages[-1]["role"] != "assistant":
    qa_chain = load_qa_chain(ChatOpenAI(model="gpt-3.5-turbo"), chain_type="stuff")
    docs = document_embeddings.similarity_search(query)
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            response = get_llm_response(query, local_chat_history, docs, qa_chain) 
            st.write(response) 
 
    st.session_state.messages.append({"role": "assistant", "content": response})
