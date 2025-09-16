import os
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

from ..core.config import URLS
from scrapping_manager import Scrap_manager

class Rag_Pipeline:
    def __init__(self, llm):
        self.llm = llm

    async def _scrapp_data(self, URLS=URLS):
        scp = Scrap_manager(URLS)
        scp.scrapp_and_save()

    async def _load_docs(self):
        docs = CSVLoader(file_path='./data/data.csv', csv_args={'delimiter':';'}).load()
        return docs

    async def _split_docs(self, docs):
        data = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
        return data

    async def _embedd(self, data):
        pass

    async def _vec_store(self, embedded_data):
        pass

    async def _retrieve(self, input):
        pass

    async def query_docs(self, input):
        if not os.path.exists('./data'):
            self._scrapp_data()
        if not os.path.exists('./chroma_db'):
            docs = self._load_docs()
            data = self._split_docs(docs)
            embedded_data = self._embedd(data)
            self._vec_store(embedded_data)
        query = self._retrieve(input)
        return query