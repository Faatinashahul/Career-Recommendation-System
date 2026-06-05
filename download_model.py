from transformers import pipeline

# Load FLAN-T5 small (lightweight, safe for MacBook)
generator = pipeline("text2text-generation", model="google/flan-t5-small")

# Test with a sample
output = generator("Suggest a career path for someone skilled in Python, AI, and data analysis.", max_length=100)
print(output[0]["generated_text"])
