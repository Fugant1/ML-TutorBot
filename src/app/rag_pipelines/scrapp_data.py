from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from bs4 import BeautifulSoup as Soup

def robust_html_extractor(html: str) -> str:
    soup = Soup(html, "lxml")
    main_content = soup.find("main")
    
    if main_content:
        return main_content.text
    else:
        return "" 

#Using user_guide to have all the important data of the site
def scrapp_data(url):
    #using max_depth 2 as a test, we don't want to load ALL the site, just the main topics, most of the queries probably will not need to deep sklearn
    loader = RecursiveUrlLoader(url=url, max_depth=2, extractor=robust_html_extractor)

    #after finishing the scrapps, I will test it :)
    return loader.load()

#my test feature for now, lol ¬¬
if __name__ == "__main__":
     docs = scrapp_data('https://scikit-learn.org/stable/user_guide.htmls')
     print(docs)