import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time
load_dotenv()

# Set the environment variables of Groq and Gemini
groq_api_key=os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")

st.title("Simple RAG")

llm=ChatGroq(groq_api_key=groq_api_key,model_name="Llama3-8b-8192")

prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)

def vector_embedding():

    if "vectors" not in st.session_state:
        # Setting the LLM to use embeddings
        st.session_state.embeddings=GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        # entering the directory which contains the PDFs
        st.session_state.loader=PyPDFDirectoryLoader("./pdf_directory")
         ## Loading documents from the Directory that was loaded previously
        st.session_state.docs=st.session_state.loader.load() 
        ## Splitting the documents into chunksfor easier access
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) 
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:20]) 
        ## Storing the documents in the FAISS vector store 
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) 


# Button to tell the app to start the vector embedding process
if st.button("Start Vector Embedding"):
    vector_embedding()
    st.write("Vector Store DB Is Ready")
input=st.text_input("Enter Your Question From Doduments")




if input:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    start=time.process_time()
    response=retrieval_chain.invoke({'input':input})
    ## to know the reponse time of process to take place
    st.write("Response time :",time.process_time()-start)
    st.write(response['answer'])
