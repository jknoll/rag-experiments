#!/usr/bin/env python3
# N.B this uses "Embedded Weaviate", where Weaviate is embedded within your application
# https://weaviate.io/developers/weaviate/installation/embedded
# data is written to ~/.local/share/weaviate unless persistence_data_path is passed as a
# configuration parameter.
import warnings
warnings.filterwarnings("ignore")

import dotenv
dotenv.load_dotenv()

import requests
# from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader

# url = "https://raw.githubusercontent.com/langchain-ai/langchain/master/docs/docs/modules/state_of_the_union.txt"
# res = requests.get(url)
# with open("state_of_the_union.txt", "w") as f:
#     f.write(res.text)

# loader = TextLoader('./state_of_the_union.txt')

loader = DirectoryLoader('logseq', glob="**/*.md", show_progress=True)
# loader = DirectoryLoader('logseq', glob="Ablation.md", show_progress=True)
documents = loader.load()
print(len(documents))

from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Weaviate
import weaviate
from weaviate.embedded import EmbeddedOptions

client = weaviate.Client(
    embedded_options = EmbeddedOptions
)

vectorstore = Weaviate.from_documents(
     client = client,
     documents = chunks,
     embedding = OpenAIEmbeddings(),
     by_text = False
)

retriever = vectorstore.as_retriever()

from langchain.prompts import ChatPromptTemplate

template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

prompt = ChatPromptTemplate.from_template(template)

print(prompt)

from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

llm = ChatOpenAI(model_name="gpt-4", temperature=0)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()   
)

print("PROMPT:\n\n")
print(prompt)

query = "What have I written about low rank adaptation, and what related concepts should I research?"
print("Output: " + rag_chain.invoke(query))