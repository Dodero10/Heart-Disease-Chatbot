import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from prepare_retrieval import get_bm25_scores


def RAG():
    # load the pdf files from the path
    loader = DirectoryLoader('retrieval/', glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()

    # split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(documents)

    # create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': "cpu"})

    # vectorstore
    vector_store = FAISS.from_documents(text_chunks, embeddings)

    # create llm
    llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q4_0.bin", model_type="llama",
                        config={'max_new_tokens': 128, 'temperature': 0.01})

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                  retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                  memory=memory)
    return chain


st.title("Heart Disease ChatBot 🧑🏽‍⚕️")
abc = None  # Khởi tạo biến abc với giá trị None

def get_abc():
    global abc  # Sử dụng biến global abc
    if abc is None:
        abc = RAG()  # Gán giá trị RAG cho biến global abc
    return abc

def conversation_chat(query):
    if len(st.session_state['history']) == 0:
        get_bm25_scores(query["content"])
        output = "What do you want to ask about " + query["content"] + "?"
        response = {"role":"assistant", "content":output}
        st.session_state['history'].append((query["content"], output))

        abc = get_abc()  # Sử dụng hàm get_abc() để gán giá trị abc từ lần đầu

        return response
    else:
        abc = get_abc()  # Sử dụng hàm get_abc() để gán giá trị abc nếu cần

        result = abc({"question": query["content"], "chat_history": st.session_state['history']})
        st.session_state['history'].append((query["content"], result["answer"]))
        response = {"role": "assistant", "content": result["answer"]}
        return response


def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role":"user", "content": "Hello!"})
        st.session_state.messages.append({"role":"assistant", "content": "Hello, which heart disease do you care about?"})

def display_chat():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("What is up")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)

        req = {"role":"user", "content":prompt}
        st.session_state.messages.append(req)

        res = conversation_chat(req)
        with st.chat_message("assistant"):
            st.markdown(res["content"])
        st.session_state.messages.append(res)

initialize_session_state()
display_chat()