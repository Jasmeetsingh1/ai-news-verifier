from transformers import pipeline
import re
from scraper import extract_article_text, get_related_articles_texts
from summarizer import summarize_text

def is_url(text):
    return re.match(r'https?://', text.strip()) is not None

def is_claim(text):
    return len(text.split()) <= 10

def is_full_article(text):
    num_words = len(text.split())
    num_sentences = text.count('.') + text.count('!') + text.count('?')
    return num_words > 50 or num_sentences > 3

def clean_query(text):
    if not text:
        return ""
    return text.strip().replace("\n", " ").replace(".", "").replace("?", "").replace("!", "")

def classify_input(text_type, text):
    if text_type == "url":
        return "url"
    elif text_type == "article":
        return "article"
    elif text_type == "claim":
        return "claim"
    else:
        return "unknown"

def process_input(input_type, user_input):
    input_type = classify_input(input_type, user_input)
    print(f"🔍 Detected input type: {input_type}")

    if input_type == "url":
        print("📰 Extracting article from URL...")
        title, article_text = extract_article_text(user_input)
        if not article_text:
            print("❌ Failed to extract main article content.")
            return "", []
        query = clean_query(article_text)
    elif input_type == "article":
        article_text = user_input
        query = clean_query(user_input)
        title = "User Provided Article"
    else:
        article_text = user_input
        query = clean_query(user_input)
        title = "User Claim"

    print("🔗 Getting related articles...")
    related_articles = get_related_articles_texts(query)

    # Filter out any articles that failed to extract
    filtered_articles = [(url, t, txt) for url, t, txt in related_articles if txt]

    print(f"\n🗞️ {len(filtered_articles)} Related Articles fetched:\n")
    for idx, (url, art_title, preview) in enumerate(filtered_articles, 1):
        print(f"🔗 Article {idx}: {url}")
        print(f"📌 Title: {art_title}")
        print(f"📝 Preview: {preview[:300]}...\n")

    print("✅ Ready for fact extraction and comparison.\n")
    return article_text, filtered_articles

def run_nli_inference(claim_or_text, related_articles):
    print("\n🔍 Running Fact Comparison...\n")
    nli_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    for idx, (url, title, content) in enumerate(related_articles, 1):
        try:
            if len(content.split()) > 400:
                content = summarize_text(content, max_length=350, min_length=40)

            result = nli_model(
                sequences=content,
                candidate_labels=["ENTAILMENT", "CONTRADICTION", "NEUTRAL"],
                hypothesis_template="{}",
                multi_label=False,
                hypothesis=claim_or_text
            )

            # Sort by confidence to get top label
            top_result = sorted(result["scores"], reverse=True)[0]
            top_label = result["labels"][result["scores"].index(top_result)]
            confidence = round(top_result * 100, 2)

            if top_label == "ENTAILMENT" and confidence >= 70:
                verdict = "✅ Supported"
            elif top_label == "CONTRADICTION" and confidence >= 70:
                verdict = "❌ Not Supported / Refuted"
            else:
                verdict = "❓ Possibly Related / Unclear"

            print(f"🔗 Article {idx}: {title}")
            print(f"🧠 Result: {verdict} (confidence: {confidence}%)\n")

        except Exception as e:
            print(f"❌ Failed to run inference for Article {idx}: {e}")

# ✅ For testing only
if __name__ == "__main__":
    user_choice = input("What are you entering? [claim/article/url]: ").strip().lower()
    user_input = input("🧾 Paste your input (URL, article or claim):\n")
    main_article, other_articles = process_input(user_choice, user_input)

    if main_article:
        print(f"\n📄 Main Article:\n{main_article[:500]}...")
        run_nli_inference(main_article, other_articles)


