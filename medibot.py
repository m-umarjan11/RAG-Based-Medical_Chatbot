import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

# Constants
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
DB_FAISS_PATH = "vectorstore/db_faiss"
HF_TOKEN = os.environ.get("HF_TOKEN")

# Set Hugging Face API token in environment
if HF_TOKEN:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN
else:
    st.error("HuggingFace API token not found in environment variables")
    st.stop()

@st.cache_resource
def get_vectorstore():
    try:
        embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error(f"Failed to load vector store: {str(e)}")
        return None

def set_custom_prompt():
    custom_prompt_template = """
    Use the pieces of information provided in the context to answer user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Don't provide anything out of the given context.

    Context: {context}
    Question: {question}

    Start the answer directly. No small talk please.
    """
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def load_llm():
    try:
        return HuggingFaceEndpoint(
            repo_id=HUGGINGFACE_REPO_ID,
            temperature=0.5,
            max_length=512  # Changed from model_kwargs to direct parameter
        )
    except Exception as e:
        st.error(f"Failed to load LLM: {str(e)}")
        return None

def main():
    st.title("🩺 Medical Chatbot")
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    if prompt := st.chat_input("Pass your prompt here"):
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Vector store not available")
                return

            llm = load_llm()
            if llm is None:
                st.error("LLM not available")
                return
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt()}
            )

            response = qa_chain.invoke({'query': prompt})
            result = response.get("result", "No answer found.")
            source_documents = response.get("source_documents", [])

            response_text = f"{result}\n\nSources:\n"
            for doc in source_documents:
                response_text += f"- {doc.metadata.get('source', 'Unknown source')} (page {doc.metadata.get('page', 'N/A')})\n"

            st.chat_message('assistant').markdown(response_text)
            st.session_state.messages.append({'role': 'assistant', 'content': response_text})

        except Exception as e:
            st.error(f"Error processing your request: {str(e)}")

if __name__ == "__main__":
    main()