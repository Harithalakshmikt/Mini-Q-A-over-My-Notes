from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

class FlanGenerator:
    def __init__(self, model_name: str = "google/flan-t5-base", max_new_tokens: int = 200):
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.pipe = pipeline("text-generation", model=model, tokenizer=tok, max_new_tokens=max_new_tokens)

    def generate(self, context: str, question: str) -> str:
        prompt =  f"""
You are a helpful assistant.

Answer the question clearly and concisely using ONLY the given context.

If the answer is not in the context, say "I don't know".\nContext:\n{context}\n\nQuestion: {question}"
        return self.pipe(prompt)[0]["generated_text"]
