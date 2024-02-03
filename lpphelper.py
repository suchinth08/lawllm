import transformers
import torch
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import Chroma
import textwrap

def gen_vectordb():
    tokenizer = AutoTokenizer.from_pretrained("lmsys/fastchat-t5-3b-v1.0")
    model = AutoModelForSeq2SeqLM.from_pretrained("lmsys/fastchat-t5-3b-v1.0")
    pipe = pipeline(
        "text2text-generation",
        model=model, 
        tokenizer=tokenizer, 
        max_length=256
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    loader = DirectoryLoader('C:/Users/SudheerRChinthala/sivallm/new_papers', glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base") 
    persist_directory = 'db'
    embedding = instructor_embeddings
    vectordb = Chroma.from_documents(documents=texts, 
                                    embedding=embedding,
                                    persist_directory=persist_directory)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=local_llm, 
                                    chain_type="stuff", 
                                    retriever=retriever, 
                                    return_source_documents=True)
    vectordb.persist()
    vectordb = None 


if __name__=="__main__":
    gen_vectordb()