import os
import streamlit as st
import torch
import transformers
import accelerate
import json
import textwrap
from typing import Set
from streamlit_chat import message
from typing import Any, List, Dict
from transformers import pipeline

from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain

from transformers import AutoTokenizer, AutoModelForCausalLM

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

def get_file_path(uploaded_file):
    cwd = os.getcwd()
    temp_dir = os.path.join(cwd, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

with st.sidebar:
    f = st.file_uploader("Upload a file", type=(["pdf"]))
    if f is not None:
        path_in = get_file_path(f)
        print("*"*10,path_in)
        st.session_state["file_uploaded"] = True
        #st.success("File uploaded successfully!")
    else:
        path_in = None
        st.session_state["file_uploaded"] = False
        #st.error("No file uploaded")
   
st.header("PDF Reader and AI-Powered Answering Bot")

prompt = st.text_input("Prompt", placeholder="Enter your prompt here..")
if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "model" not in st.session_state:    
    model_id = 'meta-llama/Llama-2-7b-chat-hf'
    hf_auth = 'xxxxxxx'
    tokenizer = AutoTokenizer.from_pretrained(model_id,
                                          token=hf_auth,)

    model = AutoModelForCausalLM.from_pretrained(model_id,
                                             trust_remote_code=True,            
                                             device_map='auto',                                             
                                             token=hf_auth,                                            
                                             )

    pipe = pipeline("text-generation",
                model=model,
                tokenizer= tokenizer,                
                max_new_tokens = 512,
                do_sample=True,
                top_k=30,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id
                )
    
    llm = HuggingFacePipeline(pipeline = pipe, model_kwargs = {'temperature':0})
    st.session_state["model"] = llm

if "vectorstore" not in st.session_state and path_in:
    loader=PyPDFLoader(file_path=path_in)
    documents=loader.load()
    text_splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=30,separator="\n")
    docs=text_splitter.split_documents(documents=documents)
    model_name = "sentence-transformers/all-mpnet-base-v2"
    hf = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore=FAISS.from_documents(docs,hf)
    vectorstore.save_local('langchain_pyloader/vectorize')
    new_vectorstore=FAISS.load_local("langchain_pyloader/vectorize", hf)
    print("pdf read done and vectorization completed.")
     
    st.session_state["vectorstore"] = new_vectorstore


if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""



def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT ):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

def cut_off_text(text, prompt):
    cutoff_phrase = prompt
    index = text.find(cutoff_phrase)
    if index != -1:
        return text[:index]
    else:
        return text

def remove_substring(string, substring):
    return string.replace(substring, "")



def generate(text):
    prompt = get_prompt(text)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
        outputs = model.generate(**inputs,
                                 max_new_tokens=512,
                                 eos_token_id=tokenizer.eos_token_id,
                                 pad_token_id=tokenizer.eos_token_id,
                                 )
        final_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        final_outputs = cut_off_text(final_outputs, '</s>')
        final_outputs = remove_substring(final_outputs, prompt)

    return final_outputs

def parse_text(text):
        wrapped_text = textwrap.fill(text, width=100)
        print(wrapped_text +'\n\n')
    
def pdf_chat(path):
    # print("pdf read done")
    pass
def ans_ret(query,new_vectorstore,chat_history):
    llm=st.session_state["model"]
    
    qa = ConversationalRetrievalChain.from_llm(
       llm=llm, retriever=new_vectorstore.as_retriever()
    )
    res=qa.invoke({"question": query, "chat_history":chat_history})
      
    return res



if prompt and path_in:
    
    with st.spinner("Generating response.."):
        
        new_vectorstore=st.session_state["vectorstore"]
        generated_response = ans_ret(
            query=prompt,new_vectorstore=new_vectorstore,chat_history=st.session_state["chat_history"]
        )
    
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(generated_response["answer"])
        st.session_state["chat_history"].append((prompt,generated_response["answer"]))
        
if st.session_state["chat_answers_history"]:
    max_len = min(len(st.session_state["chat_answers_history"]), len(st.session_state["user_prompt_history"]))
    for i in range(max_len):
        user_query = st.session_state["user_prompt_history"][i]
        gresponse = st.session_state["chat_answers_history"][i]
        message(user_query, is_user=True, key=i)
        message(gresponse, key=f"response_{i}")
