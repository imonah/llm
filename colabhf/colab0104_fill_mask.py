from transformers import pipeline

unmasker = pipeline("fill-mask", model="bert-base-cased")

# Using [MASK] instead of <mask> as that's the correct mask token for BERT
result = unmasker("This course will teach you all about [MASK] models.", top_k=3)

# Print the results in a readable format
for i, item in enumerate(result, 1):
    print(f"\nOption {i}:")
    print(f"Sequence: {item['sequence']}")
    print(f"Score: {item['score']:.4f}")
    print(f"Token: {item['token_str']} (ID: {item['token']})")