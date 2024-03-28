
import streamlit as st
import logging
from loguru import logger

from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader, TextLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS

from langchain_community.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

# 설정 로깅
logging.basicConfig(level=logging.INFO)

def initialize_session_states():
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = False
        
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]

def main():
    st.set_page_config(page_title="Rag Chat bot", page_icon=":file:")
    st.title("_File Retriever Based:red[QA Chat]_ :file:")

    initialize_session_states()

    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx', 'txt'], accept_multiple_files=True)
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
        source_documents = result['source_documents']

        st.session_state.messages.append({"role": "assistant", "content": response})
        display_source_documents(source_documents)

def display_source_documents(source_documents):
    with st.expander("참고 문서 확인"):
        for doc in source_documents:
            st.markdown(f"- {doc.metadata['source']}: {doc.page_content}")
import os 
def get_text(docs):
    doc_list = []
    for doc in docs:
        file_name = doc.name
        file_content = doc.getvalue()

        # 파일을 시스템에 저장
        temp_file_path = save_file(file_name, file_content)

        # 적절한 문서 로더를 선택하고 문서를 처리
        loader = select_document_loader(file_name, temp_file_path)
        if loader:
            documents = loader.load_and_split()
            doc_list.extend(documents)
        
       
        # 처리가 끝난 임시 파일 삭제
        os.remove(temp_file_path)

    return doc_list

def save_file(file_name, file_content):
    # 파일 확장자에 따라 인코딩 방식 결정
    if '.txt' in file_name:
        # 텍스트 파일은 UTF-8로 저장
        mode = 'w'
        content = file_content.decode('utf-8')
    else:
        # 다른 파일 유형은 바이너리 모드로 저장
        mode = 'wb'
        content = file_content

    temp_file_path = file_name  # 이 경로는 실제로 임시 파일을 저장할 위치를 반영해야 함
    with open(temp_file_path, mode) as file:
        file.write(content)

    return temp_file_path

def select_document_loader(file_name, file_path):
    if '.pdf' in file_name:
        return PyPDFLoader(file_path)
    elif '.docx' in file_name:
        return Docx2txtLoader(file_path)
    elif '.pptx' in file_name:
        return UnstructuredPowerPointLoader(file_path)
    elif '.txt' in file_name:
        return TextLoader(file_path)
    else:
        logger.error(f"Unsupported file type: {file_name}")
        return None


# 이하 함수는 상황에 따라 수정이 필요할 수 있음
def get_text_chunks(text):
    # 이 부분은 tiktoken_len 함수의 구현에 따라 다를 수 있음
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=100, length_function=len)
    chunks = text_splitter.split_documents(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask", model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
    vector_store = FAISS.from_documents(text_chunks, embeddings)
    return vector_store

def get_conversation_chain(vector_store, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-4', temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever(search_type='mmr', verbose=True), memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'), get_chat_history=lambda h: h, return_source_documents=True, verbose=True)
    return conversation_chain

if __name__ == '__main__':
    main()
