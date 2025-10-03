from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from bs4 import BeautifulSoup as Soup
import requests
import re
from urllib.parse import urljoin

def robust_html_extractor(html: str) -> str:
    soup = Soup(html, "lxml")
    important_text = soup.find("main")
    if important_text:
        soup = important_text
    return re.sub(r"\s+", " ", soup.text).strip()

#Using user_guide to have all the important data of the site
def scrapp_data(url):
    resp = requests.get(url)
    if resp.status_code != 200:
        print(f"Failed to scrapp data from {url}, status code: {resp.status_code}")
        return []
    #using max_depth 2 as a test, we don't want to load ALL the site, just the main topics, most of the queries probably will not need to deep sklearn
    loader = RecursiveUrlLoader(url=url, max_depth=2, extractor=robust_html_extractor)

    #after finishing the scrapps, I will test it :)
    return loader.load()

#my test feature for now, lol ¬¬
# if __name__ == "__main__":
#      docs = scrapp_data('https://scikit-learn.org/stable/user_guide')
#      print(docs)