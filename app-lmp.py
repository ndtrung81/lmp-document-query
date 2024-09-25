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
    index = load_index_from_storage(storage_context,embed_model=embed_model,
                                            transformations=transformations)


#sample_questions = generate_sample_questions(documents)
query_engine = CitationQueryEngine.from_args(
    index,
    llm=llm,
    similarity_top_k=3,
    streaming=True,
    # here we can control how granular citation sources are, the default is 512
    citation_chunk_size=512,
)

#response = query_engine.query("list the kind of gpus that are available in the cluster")
#response = query_engine.query("What is pair style lj/cut?")
# response = query_engine.query("which node has the most RAM in the cluster")
# response = query_engine.query("write a sample batch script for me. my python file is hello_world.py")
# response = query_engine.query("how can write a job that uses 2 gpus and 4 cpus. Write a sample script for that")
# <---  Include the whole existing python script here --->

st.markdown("### 📘 LAMMPS User Guide Chatbot 🤖")
# write more description here
st.markdown(
    """ This chatbot sources its information from the [LAMMPS Documentation](https://docs.lammps.org/Manual.html)"""
)


with st.expander("🤔 FAQs :", expanded=False):
    st.markdown(
        """
            - What is pair style lj/cut?
            - How can I use compute pe/atom?
            - Write an input script to use fix ave/correlate
            """
    )

# The input
search_input = st.text_input("Enter your query here:")
if search_input:
    # Query the model
    streaming_response = query_engine.query(search_input)
    res_box = st.empty()
    report = []
    for word in streaming_response.response_gen:
        # show output without new line
        report.append(word)
        result = "".join(report).strip()
        # result = result.replace("\n", "")
        res_box.markdown(f"{result}")
    # Mardown heading for References
    st.markdown("### References")
    list_url = []

    unique_source_nodes = {}
    for source_node in streaming_response.source_nodes:
        # Using node_id as a unique identifier
        if source_node.node_id not in unique_source_nodes:
            unique_source_nodes[source_node.node_id] = source_node

    # Print the references
    for ref in streaming_response.source_nodes:
        source_path = ref.node.metadata["file_path"]
        # remove .md from the path
        #source_path = source_path[:-3]
        # remove prefix docs from the path
        #source_path = source_path[5:]
        # add the base url
        #source_path = github_url + source_path
        # show the souce path url on streamlit ui
        list_url.append(source_path)

    # remove duplicates but keep the order of the list
    # list_url = list(dict.fromkeys(list_url))

    # show the souce path url on streamlit ui as link
    count = 0
    for url in list_url:
        count+=1
        st.markdown('['+str(count)+'] '+url, unsafe_allow_html=True)
    #with st.expander("🤔 Some other questions you can ask:", expanded=True):
    #    st.markdown(display_questions(sample_questions))

#st.markdown("""Made by the LAMMPS development team""")
