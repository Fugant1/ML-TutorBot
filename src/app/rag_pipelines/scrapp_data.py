from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from bs4 import BeautifulSoup as Soup

#Using user_guide to have all the important data of the site
async def scrapp_data(url):
    #using max_depth 2 as a test, we don't want to load ALL the site, just the main topics, most of the queries probably will not need to deep sklearn
    loader = RecursiveUrlLoader(url=url, max_depth=2, extractor=lambda html: Soup(html, "lxml").find("main").text)

    #after finishing the scrapps, I will test it :)
    return loader.load()

#my test feature for now, lol ¬¬
# if __name__ == "__main__":
#     docs = scrapp_data()
#     print(docs)