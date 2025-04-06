from transformers import BartForConditionalGeneration, BartTokenizer
import torch

# Load the pre-trained BART model and tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

def summarize_text(text, max_length=150, min_length=30):
    inputs = tokenizer.batch_encode_plus([text], return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs["input_ids"],
                                 num_beams=4,
                                 length_penalty=2.0,
                                 max_length=max_length,
                                 min_length=min_length,
                                 early_stopping=True)
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Example usage
if __name__ == "__main__":
    article = """India's Supreme Court on Wednesday ruled that electoral bonds, a form of anonymous political donations, are unconstitutional..."""
    print("Summary:\n", summarize_text(article))