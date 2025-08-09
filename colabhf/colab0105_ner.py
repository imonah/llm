from transformers import pipeline

# Initialize the NER pipeline
ner = pipeline("ner", aggregation_strategy="simple", model="dslim/bert-base-NER")

# Process the text
text = "My name is Sylvain and I work at Hugging Face in Brooklyn."
result = ner(text)

# Group subword tokens into complete words
grouped_entities = []
current_entity = None

for entity in result:
    word = entity['word']
    
    # If the word starts with ##, it's a continuation of the previous word
    if word.startswith('##'):
        if current_entity:
            current_entity['word'] += word[2:]
            current_entity['end'] = entity['end']
            # Update score to be the average
            current_entity['score'] = (current_entity['score'] + entity['score']) / 2
    else:
        if current_entity:
            grouped_entities.append(current_entity)
        current_entity = entity.copy()

# Add the last entity if it exists
if current_entity:
    grouped_entities.append(current_entity)

# Print the result in the desired format
print('[')
for i, entity in enumerate(grouped_entities):
    print(f" {{'entity_group': '{entity['entity_group']}', 'score': {entity['score']:.5f}, 'word': '{entity['word']}', 'start': {entity['start']}, 'end': {entity['end']}}}" + (',' if i < len(grouped_entities)-1 else ''))
print(']')