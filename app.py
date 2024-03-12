import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import docx
import docx2txt  

load_dotenv()
os.getenv("GOOGLE_API_KEY")
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_docx_text(docx_docs):
    text = ""
    for docx_file in docx_docs:
        doc = docx.Document(docx_file)
        for paragraph in doc.paragraphs:
            text += paragraph.text
    return text

def get_doc_text(doc_files):
    text = ""
    for doc_file in doc_files:
        text += docx2txt.process(doc_file)
    return text

def get_txt_text(txt_files):
    text = ""
    for txt_file in txt_files:
        if hasattr(txt_file, "read"):
            text += txt_file.read().decode("utf-8")
        else:
            with open(txt_file, "r", encoding="utf-8") as txt:
                text += txt.read()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config("Document Summarizer & Chatting")
    st.header("Chat with various File types using Gemini")

    user_question = st.text_input("Ask a Question from the any of the uploaded files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        uploaded_files = st.file_uploader("Upload your Documents", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = ""
                for uploaded_file in uploaded_files:
                    if uploaded_file.name.endswith(".pdf"):
                        raw_text += get_pdf_text([uploaded_file])
                    elif uploaded_file.name.endswith(".docx"):
                        raw_text += get_docx_text([uploaded_file])
                    elif uploaded_file.name.endswith(".doc"):
                        raw_text += get_doc_text([uploaded_file])
                    elif uploaded_file.name.endswith(".txt"):
                        raw_text += get_txt_text([uploaded_file])

                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
