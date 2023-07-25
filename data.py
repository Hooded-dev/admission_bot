import chromadb
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import UnstructuredURLLoader
from langchain.document_loaders import SeleniumURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
import os
import openai
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import inspect
import pickle
from getpass import getpass
from langchain import OpenAI
from langchain.chains import LLMChain, ConversationChain
from langchain.chains.conversation.memory import (ConversationBufferMemory,
                                                  ConversationSummaryMemory,
                                                  ConversationBufferWindowMemory,
                                                  ConversationKGMemory)
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.callbacks import get_openai_callback
from langchain.document_loaders import WebBaseLoader
import tiktoken
import os
from langchain.vectorstores import FAISS
import urllib.request
from bs4 import BeautifulSoup
import re
import requests
from urllib.parse import urljoin

urls = ['https://au.edu.pk/']

def is_admission_link(link):
    keyword_pattern = re.compile(r'Admission|admissions|Faculties', re.IGNORECASE)
    return bool(keyword_pattern.search(link))

def extract_links(urls):
    all_links = []
    for link in urls:
        response = requests.get(link)
        soup = BeautifulSoup(response.text, "html.parser")
        base_url = response.url  # Get the base URL from the response object
        links = []
        for a_tag in soup.find_all("a"):
            href = a_tag.get("href")
            if href:
                absolute_link = urljoin(base_url, href)  # Convert relative link to absolute link
                if is_admission_link(absolute_link):
                    links.append(absolute_link)
        all_links.extend(links)
    print(len(all_links))
    return all_links
def links_embed(all_links):
    loader = WebBaseLoader(all_links)
    scrape_data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter (chunk_size=1000, chunk_overlap=200)
    texts_from_links = text_splitter.split_documents(scrape_data)
    for document in texts_from_links:
        document.page_content = document.page_content.replace('\n', ' ')
    return texts_from_links

def vector_db(texts_from_links):
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")
    db_links = FAISS.from_documents(texts_from_links, embeddings)
    with open("links_db.pkl", "wb") as f:
        pickle.dump(db_links, f)