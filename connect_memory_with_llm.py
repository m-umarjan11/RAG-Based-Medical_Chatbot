import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Step 1: Setup LLM (Mistral with HuggingFace)
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# Ensure the Hugging Face API token is set
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

def load_llm(huggingface_repo_id):
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"max_length": 512}  # Ensure max_length is an integer
    )

# Step 2: Custom Prompt Template
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don’t know the answer, just say that you don’t know. Don’t try to make up an answer. 
Do not provide anything outside of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk.
"""

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

# Step 3: Load FAISS Vector Store
DB_FAISS_PATH = "vectorstore/db_faiss"

# Ensure FAISS database exists
if not os.path.exists(DB_FAISS_PATH):
    raise FileNotFoundError(f"FAISS database not found at {DB_FAISS_PATH}")

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Step 4: Create QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Step 5: Invoke with a Single Query
user_query = input("Write Query Here: ")
response = qa_chain.invoke({'query': user_query})  # Fixed: Using 'query' as required

# Print results
print("\nRESULT:", response.get("result", "No answer found."))
print("\nSOURCE DOCUMENTS:", response.get("source_documents", "No source documents found."))
