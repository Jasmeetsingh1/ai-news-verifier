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
    if text_type in ["claim", "url", "article"]:
        return text_type
    if is_url(text):
        return "url"
    elif is_claim(text):
        return "claim"
    elif is_full_article(text):
        return "article"
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
            return "", [], "", ""
        query = clean_query(article_text)

    elif input_type == "article":
        article_text = user_input
        title = "User Provided Article"
        query = clean_query(user_input)

    else:  # claim
        article_text = user_input
        title = "User Claim"
        query = clean_query(user_input)

    print("🔗 Getting related articles...")
    related_articles = get_related_articles_texts(query)
    filtered_articles = [(url, t, txt) for url, t, txt in related_articles if txt]

    print(f"\n🗞️ {len(filtered_articles)} Related Articles fetched:\n")
    for idx, (url, art_title, preview) in enumerate(filtered_articles, 1):
        print(f"🔗 Article {idx}: {url}")
        print(f"📌 Title: {art_title}")
        print(f"📝 Preview: {preview[:300]}...\n")

        # ⬇️ Summarize if it's a full article
    if input_type in ["url", "article"]:
        claim_summary = summarize_text(article_text, max_length=350, min_length=40)
        print(f"\n🧾 Extracted Summary from Article:\n{claim_summary}\n")
    else:
        claim_summary = article_text
        print(f"\n🧾 Using Claim Directly:\n{claim_summary}\n")

    return article_text, filtered_articles, input_type, claim_summary


def run_nli_inference(claim_text, related_articles):
    print("\n🔍 Running Fact Comparison...\n")
    nli_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    verdicts = []

    for idx, (url, title, content) in enumerate(related_articles, 1):
        try:
            if len(content.split()) > 400:
                content = summarize_text(content, max_length=350, min_length=40)

            result = nli_model(
                sequences=content,
                candidate_labels=["ENTAILMENT", "CONTRADICTION", "NEUTRAL"],
                hypothesis_template="{}",
                multi_label=False,
                hypothesis=claim_text
            )

            top_idx = result["scores"].index(max(result["scores"]))
            top_label = result["labels"][top_idx]
            confidence = round(result["scores"][top_idx] * 100, 2)

            if top_label == "ENTAILMENT" and confidence >= 70:
                verdict = "✅ Supported"
            elif top_label == "CONTRADICTION" and confidence >= 70:
                verdict = "❌ Not Supported / Refuted"
            else:
                verdict = "❓ Possibly Related / Unclear"

            verdicts.append({
                "url": url,
                "title": title,
                "label": top_label,
                "confidence": confidence,
                "verdict": verdict
            })

            print(f"🔗 Article {idx}: {title}")
            print(f"🧠 Result: {verdict} (confidence: {confidence}%)\n")

        except Exception as e:
            print(f"❌ Failed to run inference for Article {idx}: {e}")

    return verdicts

def final_verdict(verdicts):
    support = sum(1 for v in verdicts if v['label'] == 'ENTAILMENT' and v['confidence'] >= 70)
    refute = sum(1 for v in verdicts if v['label'] == 'CONTRADICTION' and v['confidence'] >= 70)

    if support > refute:
        return "✅ Likely True"
    elif refute > support:
        return "❌ Likely False"
    else:
        return "❓ Unclear or Inconclusive"

# ✅ For testing only
if __name__ == "__main__":
    user_choice = input("What are you entering? [claim/article/url/auto]: ").strip().lower()
    user_input = input("🧾 Paste your input (URL, article or claim):\n")

    main_article, related_articles, input_type, claim_summary = process_input(user_choice, user_input)

    if main_article and related_articles:
        print(f"\n🧾 Extracted Claim for Fact-Check:\n{claim_summary}\n")
        verdicts = run_nli_inference(claim_summary, related_articles)
        conclusion = final_verdict(verdicts)
        print(f"\n🔚 Final Verdict: {conclusion}")
