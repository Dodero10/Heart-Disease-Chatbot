import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms.ctransformers import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from streamlit_option_menu import option_menu

from prepare_retrieval import get_bm25_scores
import pymongo
database_name = "chatbot"

# URI connection string
uri = f"mongodb+srv://hoangdinhhung20012003:hust20210399@cluster0.25m2gdf.mongodb.net/{database_name}?retryWrites=true&w=majority"

# Kết nối đến MongoDB Atlas
client = pymongo.MongoClient(uri)

# Chọn database
db = client[database_name]
my_collection = db["history"]

model_dict = {"llama-2-7b": "llama-2-7b-chat.ggmlv3.q4_0.bin", "zephyr-7b": "zephyr-7b-alpha.Q4_0.gguf","mistral-7b": "mistral-7b-v0.1.Q3_K_M.gguf"}
def initialize_sidebar():
    st.sidebar.title("🤗💬 LLM Chat App about heart disease")
    st.sidebar.markdown(
        "<a href='https://github.com/Dodero10' style='color: white; text-decoration: none; font-weight: bold;'>Trương Công Đạt - 20215346</a>",
        unsafe_allow_html=True)
    st.sidebar.markdown(
        "<a href='https://github.com/phuccodetrau' style='color: white; text-decoration: none; font-weight: bold;'>Nguyễn Hoàng Phúc - 20215452</a>",
        unsafe_allow_html=True)
    st.sidebar.markdown(
        "<a href='https://github.com/hunghd20012003' style='color: white; text-decoration: none; font-weight: bold;'>Hoàng Đình Hùng - 20210399</a>",
        unsafe_allow_html=True)


def setup_RAG(model_name):
    loader = DirectoryLoader('retrieval/', glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    text_chunks = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': "cpu"})
    vector_store = FAISS.from_documents(text_chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    if model_name == "llama-2-7b-chat.ggmlv3.q4_0.bin":
        llm = CTransformers(model=model_name, model_type = "llama", config={'max_new_tokens': 128, 'temperature': 0.01})
    else:
        llm = CTransformers(model=model_name, config={'max_new_tokens': 128, 'temperature': 0.01})

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff', retriever=retriever, memory=memory)


def get_chain(model_name):
    if 'rag_chain' not in st.session_state:
        st.session_state.rag_chain= setup_RAG(model_name)
    return st.session_state.rag_chain


def handle_conversation(query, model_name):
    if len(st.session_state['history']) == 0:
        get_bm25_scores(query["content"])
    #get_bm25_scores(query["content"])
    chain = get_chain(model_name)
    result = chain({"question": query["content"], "chat_history": st.session_state['history']})
    output = result["answer"]
    st.session_state['history'].append((query["content"], output))
    # print(rt.invoke(query["content"])[0].page_content)
    # print(rt.invoke(query["content"])[1].page_content)
    return {"role": "assistant", "content": output}


def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if "messages" not in st.session_state:
        all_documents = list(my_collection.find())
        st.session_state.messages=all_documents
        st.session_state.messages.append({"role": "user", "content": "Hello!"})
        st.session_state.messages.append({"role": "assistant", "content": "Hello, which heart disease do you care about?"})
        my_collection.insert_one({"role": "user", "content": "Hello!"})
        my_collection.insert_one({"role": "assistant", "content": "Hello, which heart disease do you care about?"})

def display_chat(model_name):
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    prompt = st.chat_input("What is up")
    if prompt:
        user_message = {"role": "user", "content": prompt}
        my_collection.insert_one(user_message)
        st.session_state.messages.append(user_message)
        with st.chat_message(user_message["role"]):
            st.markdown(user_message["content"])

        res = handle_conversation(user_message, model_name)
        my_collection.insert_one(res)
        st.session_state.messages.append(res)
        with st.chat_message(res["role"]):
            st.markdown(res["content"])


initialize_sidebar()
st.title("Heart Disease ChatBot 🧑🏽‍⚕️")
selected = option_menu(menu_title=None, options=["llama-2-7b", "mistral-7b", "zephyr-7b"], default_index=0, orientation="horizontal")
initialize_session_state()
model_name = model_dict[selected]
display_chat(model_name)

