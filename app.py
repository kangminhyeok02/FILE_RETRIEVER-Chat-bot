import streamlit as st
import logging
from loguru import logger

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    TextLoader,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

# (필요하다면) from langchain.callbacks import get_openai_callback

# 로깅 설정
logging.basicConfig(level=logging.INFO)

def initialize_session_states():
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = False
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [
            {"role": "assistant", "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}
        ]

def main():
    st.set_page_config(page_title="RAG Chat Bot", page_icon=":file:")
    st.title("_File Retriever Based_ :red[QA Chat]_ :file:")

    initialize_session_states()

    with st.sidebar:
        uploaded_files = st.file_uploader(
            "Upload your file",
            type=['pdf', 'docx', 'pptx', 'txt'],
            accept_multiple_files=True
        )
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        process = st.button("Process")

    if process and openai_api_key:
        files_text = get_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        vector_store = get_vector_store(text_chunks)

        st.session_state.conversation = get_conversation_chain(vector_store, openai_api_key)
        st.session_state.processComplete = True
    elif process and not openai_api_key:
        st.error("OpenAI API 키를 입력해주세요.")

    display_messages()
    handle_user_input()

def display_messages():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def handle_user_input():
    query = st.chat_input("질문을 입력해주세요.")
    if query and st.session_state.conversation:
        process_query(query)
    elif query:
        st.error("문서를 먼저 업로드하고 'Process' 버튼을 클릭해주세요.")

def process_query(query):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.spinner("Thinking..."):
        result = st.session_state.conversation({"question": query})
        response = result['answer']
        sources = result.get('source_documents', [])

        st.session_state.messages.append({"role": "assistant", "content": response})
        display_source_documents(sources)

def display_source_documents(source_documents):
    with st.expander("참고 문서 확인"):
        for doc in source_documents:
            st.markdown(f"- **{doc.metadata.get('source','')}**: {doc.page_content}")

import os
def get_text(docs):
    doc_list = []
    for doc in docs:
        file_name = doc.name
        content = doc.getvalue()
        temp_path = save_file(file_name, content)

        loader = select_document_loader(file_name, temp_path)
        if loader:
            documents = loader.load_and_split()
            doc_list.extend(documents)

        os.remove(temp_path)
    return doc_list

def save_file(file_name, file_content):
    mode = 'wb' if not file_name.lower().endswith('.txt') else 'w'
    data = file_content if mode=='wb' else file_content.decode('utf-8')
    with open(file_name, mode) as f:
        f.write(data)
    return file_name

def select_document_loader(file_name, file_path):
    ext = file_name.lower().split('.')[-1]
    if ext == 'pdf':
        return PyPDFLoader(file_path)
    elif ext == 'docx':
        return UnstructuredWordDocumentLoader(file_path)
    elif ext == 'pptx':
        return UnstructuredPowerPointLoader(file_path)
    elif ext == 'txt':
        return TextLoader(file_path)
    else:
        logger.error(f"Unsupported file type: {file_name}")
        return None

def get_text_chunks(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=len
    )
    return splitter.split_documents(docs)

def get_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True},
    )
    return FAISS.from_documents(chunks, embeddings)

def get_conversation_chain(vector_store, openai_api_key):
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name='gpt-4',
        temperature=0,
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_type='mmr', verbose=True),
        memory=ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            output_key='answer'
        ),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )

if __name__ == '__main__':
    main()
