from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class FlanGenerator:
    def __init__(self, model_name: str = "google/flan-t5-base", max_new_tokens: int = 250):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.max_new_tokens = max_new_tokens

    def generate(self, context: str, question: str) -> str:
        prompt = f"""You are a helpful assistant. Using ONLY the given context, explain the answer to the question in 2-3 complete sentences, in your own words. If the answer is not in the context, say "I don't know".


Context: {context}

Question: {question}

Answer:
"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.strip()