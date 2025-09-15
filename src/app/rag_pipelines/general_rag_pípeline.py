class Rag_Pipeline:
    def __init__(self, llm, vectorstore, retriever):
        self.llm = llm