import logging
import sys
import os

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SentenceSplitter

from llama_index.core.query_engine import CitationQueryEngine
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)


#from backend import generate_sample_questions, display_questions
import pickle
import os
import openai
#from llama_index import download_loader
import nest_asyncio
import streamlit as st

# The nest_asyncio module enables the nesting of asynchronous functions within an already running async loop.
# This is necessary because Jupyter notebooks inherently operate in an asynchronous loop.
# By applying nest_asyncio, we can run additional async functions within this existing loop without conflicts.
nest_asyncio.apply()

# base url
#github_url = "https://github.com/lammps/lammps/tree/develop/doc"
#download_loader("GithubRepositoryReader")

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


openai.api_key = os.getenv("OPENAI_API_KEY")

required_exts = [".html"]
loader = SimpleDirectoryReader(input_dir="/home/ndtrung/Codes/lammps-git/doc/html",
                               required_exts=required_exts,
                               )

using_openai = True
if using_openai == True:
    embed_model = OpenAIEmbedding(model="text-embedding-3-small", embed_batch_size=8)
    llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo")
    PERSIST_DIR = "./citation-lmp-openai"
else:
    # bge-base embedding model
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    # ollama
    llm = Ollama(model="llama3", request_timeout=360.0)
    PERSIST_DIR = "./citation-lmp-hgf"

transformations = [SentenceSplitter(chunk_size=1024)]

if not os.path.exists(PERSIST_DIR):
    
    #documents = loader.load_data(branch="develop")
    print("loading data")
    documents = loader.load_data()
    print("done loading data")
    print(f"Loaded {len(documents)} docs")

    index = VectorStoreIndex.from_documents(documents,
                                            embed_model=embed_model,
                                            transformations=transformations) 
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

query_engine = CitationQueryEngine.from_args(
    index,
    llm=llm,
    similarity_top_k=3,
    streaming=True,
    citation_chunk_size=512,
)

#query_engine = index.as_query_engine()
response = query_engine.query("What is pair style lj/cut?")
print(response)



