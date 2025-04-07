from googlesearch import search
from newspaper import Article, Config
import time

def search_google_news(query, num_results=5):
    search_results = []
    query += " site:indiatimes.com OR site:thehindu.com OR site:ndtv.com OR site:timesofindia.indiatimes.com"
    for i, result in enumerate(search(query)):
        if i >= num_results:
            break
        search_results.append(result)
    return search_results

def extract_article_text(url):
    try:
        config = Config()
        config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
        article = Article(url, config=config)
        article.download()
        article.parse()
        return article.title, article.text
    except Exception as e:
        print(f"❌ Failed to extract article from {url}: {e}")
        return None, None

def get_related_articles_texts(query, num_results=5):
    urls = search_google_news(query, num_results)
    articles = []
    for url in urls:
        title, text = extract_article_text(url)
        if text:
            articles.append((url, title, text))
        time.sleep(1)
    return articles

# ✅ Test block
if __name__ == "__main__":
    query = input("🔍 Enter a search query: ")
    print(f"\n📡 Searching for articles related to: {query}\n")
    articles = get_related_articles_texts(query)

    print(f"\n🗞️ {len(articles)} articles fetched.\n")
    for idx, (url, title, text) in enumerate(articles, 1):
        print(f"🔗 Article {idx}: {url}")
        print(f"📌 Title: {title}")
        print(f"📝 Preview: {text[:300]}...\n")
