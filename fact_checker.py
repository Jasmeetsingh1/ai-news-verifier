from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from factextractor import extract_facts_from_text, extract_text_from_url
from scraper import process_input
import torch

# Load models
device = 0 if torch.cuda.is_available() else -1
nli_model = pipeline("text-classification", model="microsoft/deberta-large-mnli", device=device)
semantic_model = SentenceTransformer("intfloat/e5-large-v2", device=device)

def filter_by_semantic_similarity(claims, evidence_sentences, threshold=0.6, top_k=3):
    """Filter evidence by semantic similarity to the claim."""
    results = []
    for claim in claims:
        claim_embed = semantic_model.encode(claim, convert_to_tensor=True)
        evidence_embed = semantic_model.encode(evidence_sentences, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(claim_embed, evidence_embed)[0]
        top_results = torch.topk(scores, k=min(top_k, len(scores)))
        selected = [(evidence_sentences[i], scores[i].item()) for i in top_results.indices if scores[i] >= threshold]
        results.append((claim, selected))
    return results

def fact_check(claims, related_articles, sim_threshold=0.6):
    verdicts = []

    for url, title in related_articles:
        content = extract_text_from_url(url)
        if not content:
            continue

        evidence = extract_facts_from_text(content, num_facts=5)
        semantic_filtered = filter_by_semantic_similarity(claims, evidence, threshold=sim_threshold)

        for claim, similar_evidences in semantic_filtered:
            for evidence_text, sim_score in similar_evidences:
                try:
                    result = nli_model(f"{claim} </s> {evidence_text}")[0]
                    label = result["label"]
                    score = round(result["score"] * 100, 2)

                    if label == "ENTAILMENT" and score >= 70:
                        verdict = "✅ Supported"
                    elif label == "CONTRADICTION" and score >= 70:
                        verdict = "❌ Refuted"
                    else:
                        verdict = "❓ Unclear"

                    verdicts.append({
                        "claim": claim,
                        "evidence": evidence_text,
                        "similarity": round(sim_score * 100, 2),
                        "url": url,
                        "source_title": title,
                        "verdict": verdict,
                        "confidence": score
                    })
                except Exception as e:
                    print(f"NLI failed: {e}")
    return verdicts

def compute_final_verdict(verdicts):
    supported = [v for v in verdicts if v["verdict"] == "✅ Supported"]
    refuted = [v for v in verdicts if v["verdict"] == "❌ Refuted"]

    if len(supported) > len(refuted):
        avg_conf = sum(v["confidence"] for v in supported) / len(supported)
        return "✅ Likely True", min(95, max(50, int(avg_conf)))
    elif len(refuted) > len(supported):
        avg_conf = sum(v["confidence"] for v in refuted) / len(refuted)
        return "❌ Likely False", min(95, max(20, int(avg_conf)))
    else:
        return "❓ Unclear", 45

if __name__ == "__main__":
    user_input = input("Paste claim, article, or URL:\n").strip()
    related_articles = process_input(user_input)

    if not related_articles:
        print("❌ Could not fetch related articles.")
        exit()

    # Limit to top 3 related articles
    related_articles = related_articles[:3]

    if user_input.startswith("http"):
        input_text = extract_text_from_url(user_input)
    elif len(user_input) < 50:
        input_text = user_input
    else:
        input_text = user_input

    claims = extract_facts_from_text(input_text, num_facts=3 if len(user_input) > 50 else 1)
    print("\n🔍 Extracted Claims:")
    for i, c in enumerate(claims, 1):
        print(f"{i}. {c}")

    verdicts = fact_check(claims, related_articles)
    final_verdict, credibility = compute_final_verdict(verdicts)

    print("\n📊 Fact-Check Summary:")
    for v in verdicts:
        print(f"🧠 Claim: {v['claim']}")
        print(f"📜 Evidence: {v['evidence']}")
        print(f"🔗 Source: {v['url']}")
        print(f"🤖 Verdict: {v['verdict']} ({v['confidence']}%)\n")

    print(f"🔚 Final Verdict: {final_verdict}")
    print(f"📊 Credibility Score: {credibility}/100")
