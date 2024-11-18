import streamlit as st
from llama_index.core import VectorStoreIndex, Document
from llama_index.llms.openai import OpenAI
import openai
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Settings

openai.api_key = st.secrets['OPEN_AI_TOKEN']
st.header("Converse com a gente 💬 📚")

if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Me faça uma pergunta sobre os imóveis da incorporadora ICN3"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Carregando e indexando os Documentos! Isso pode levar alguns minutos."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0, system_prompt="Você é um vendedor da incorporadora ICN3 e seu trabalho é responder perguntas sobre essa incorporadora e realizar a venda dos imóveis da incorporadora. Seja cordeal e mantenha suas respostas baseadas em fatos, não tenha alucinações")
        index = VectorStoreIndex.from_documents(docs, show_progress=True)
        return index

index = load_data()

chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.chat_input("Sua pergunta"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history
