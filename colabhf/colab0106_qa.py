from transformers import pipeline

question_answerer = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
result = question_answerer(
    question="Where do I work?",
    context="My name is Sylvain and I work at Hugging Face in Brooklyn",
)

# Print the result in the desired format
print(result)