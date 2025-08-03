from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")

# Generate text with explicit parameters
results = generator(
    "In this course, we will teach you how to",
    max_new_tokens=30,  # Use either max_new_tokens or max_length, not both
    num_return_sequences=2, # Number of sequences to generate
    truncation=True,    # Explicitly enable truncation
    pad_token_id=50256  # Explicitly set pad_token_id to eos_token_id
)

# Print the generated texts
for i, result in enumerate(results, 1):
    print(f"\nGenerated text {i}:")
    print(result['generated_text'])