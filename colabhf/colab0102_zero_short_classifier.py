from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
result = classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)

print("Результаты классификации:")
for label, score in zip(result['labels'], result['scores']):
    print(f"{label}: {score:.2f}")

# Альтернативный вариант с процентами:
print("\nВероятности в процентах:")
for label, score in zip(result['labels'], result['scores']):
    print(f"{label}: {score*100:.1f}%")