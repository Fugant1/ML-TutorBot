import pandas as pd
from src.app.rag_pipelines.scrapp_data import scrapp_data

class Scrap_manager():
    def __init__(self, URLS):
        self.urls = URLS

    async def _scrapp_all(self):
        docs = []
        for url in self.urls:
            if url != '':
                loaded_doc = scrapp_data(url)
                docs.extend(loaded_doc)
                print(f"Scrapped {url} with {len(loaded_doc)} documents.")
        return docs
    
    async def scrapp_and_save(self):
        docs = await self._scrapp_all()
        data_to_df = [
            {
                'text': doc.page_content,
                'source': doc.metadata.get('source', 'N/A')
            }
            for doc in docs
        ]
        df = pd.DataFrame(data_to_df)
        df.to_csv('data/data.csv', index=False)