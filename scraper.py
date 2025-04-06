from googlesearch import search
from newspaper import Article
import time

def search_google_news(query, num_results=5):
    search_results = []
    query += " site:indiatimes.com OR site:thehindu.com OR site:ndtv.com"
    for i, result in enumerate(search(query)):
        if i >= num_results:
            break
        search_results.append(result)
    return search_results

def extract_article_text(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"❌ Failed to extract article from {url}: {e}")
        return None

if __name__ == "__main__":
    query = "India electoral bonds Supreme Court"
    urls = search_google_news(query)
    print("🔗 Found URLs:", urls)

    for url in urls:
        print(f"\n📰 Article from: {url}")
        print(extract_article_text(url))
        time.sleep(1)
