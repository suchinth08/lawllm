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
import streamlit as st

persist_directory = 'db'
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base") 
embedding = instructor_embeddings
#tokenizer = AutoTokenizer.from_pretrained("lmsys/fastchat-t5-3b-v1.0")
tokenizer = AutoTokenizer.from_pretrained("lmsys/fastchat-t5-3b-v1.0",use_fast=False, legacy=False)
model = AutoModelForSeq2SeqLM.from_pretrained("lmsys/fastchat-t5-3b-v1.0")
pipe = pipeline("text2text-generation",model=model, tokenizer=tokenizer)
local_llm = HuggingFacePipeline(pipeline=pipe)
vectordb = Chroma(persist_directory=persist_directory,embedding_function=embedding)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

def get_lpphelper_chain():
    qa_chain = RetrievalQA.from_chain_type(llm=local_llm, 
                                  chain_type="stuff", 
                                  retriever=retriever, 
                                  return_source_documents=True)
    return qa_chain

def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

def process_llm_response(llm_response):
    wrap_text = wrap_text_preserve_newlines(llm_response['result'])   
    sources = '\n\nSources:'
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        sources.join(source.metadata['source'])  
    print(wrap_text.join(sources))    
    return wrap_text.replace("<pad>","")

if __name__=="__main__":
    get_lpphelper_chain()