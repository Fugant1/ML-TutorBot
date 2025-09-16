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
    def __init__(self):
        self.embedded_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    async def _scrapp_data(self, URLS=URLS):
        #throw the urls to the scrap manager to get all the data and save it in a csv :)
        scp = Scrap_manager(URLS)
        scp.scrapp_and_save()

    async def _load_docs(self):
        #pick all the data and load it to be processed
        docs = await CSVLoader(file_path='./data/data.csv', csv_args={'delimiter':';'}).aload()
        return docs

    async def _split_docs(self, docs):
        #just splits the daa in chunks
        data = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
        return data

    async def _embedd_and_vec_store(self, splited_data):
        #just embedds
        await Chroma.afrom_documents(documents=splited_data, embedding=self.embedded_model, persist_directory="./chroma_db")

    async def _retrieve(self, input):
        #retrieve the most relevant data
        vector_store = Chroma(persist_directory="./chroma_db", embedding_function=self.embedded_model)
        retrieve = vector_store.as_retriever()
        retrieved = await retrieve.ainvoke(input)
        return retrieved

    async def query_docs(self, input):
        #abstraction to call all the internal steps, just runs the pipeline and returns the query, aka the relevant docs
        if not os.path.exists('./data'):
            await self._scrapp_data()
        if not os.path.exists('./chroma_db'):
            docs = await self._load_docs()
            splited_data = await self._split_docs(docs)
            await self._embedd_and_vec_store(splited_data)
        query = await self._retrieve(input)
        return query