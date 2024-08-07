import pdfplumber
import spacy
from spacy.tokens import DocBin
from spacy.training import Example
import random
from spacy import displacy
from spacy.training import offsets_to_biluo_tags


pdf_path = r'C:\Users\chant\OneDrive\Escritorio\TD8\Modern Architecture_CH2020.pdf'

##Extrae las palabras de una pagina dada
try:
    with pdfplumber.open(pdf_path) as pdf:
        # Check the number of pages
        num_pages = len(pdf.pages)
        print(f'The PDF has {num_pages} pages.')

        p =18
        _page = pdf.pages[p]
        text = _page.extract_text()

        # Si el texto es nulo, printea eso
        if text is None:
            print("No text found on the page.")
        else:
            print(f'Text on the p page:\n{text}')
except Exception as e:
    print(f'An error occurred: {e}')


# Se define un conjunto de training data
TRAIN_DATA = [
    ("One of the most famous examples of modern architecture is the Villa Savoye by Le Corbusier.", {"entities": [(57, 69, "ARCH")]}),
    ("Another notable work is Fallingwater by Frank Lloyd Wright.", {"entities": [(24, 35, "ARCH")]}),
    ("The Guggenheim Museum, designed by Frank Lloyd Wright, is a masterpiece of modern architecture.", {"entities": [(4, 23, "ARCH")]}),
    ("The Sydney Opera House is an iconic example of modern architecture.", {"entities": [(4, 22, "ARCH")]}),
]
# Carga el modelo 
nlp = spacy.blank("en")

def check_alignment(text, entities):
    doc = nlp.make_doc(text)
    biluo_tags = offsets_to_biluo_tags(doc, entities)
    return biluo_tags

# Verifica la alineación 
for text, annotations in TRAIN_DATA:
    entities = annotations.get("entities")
    biluo_tags = check_alignment(text, entities)
    print(f'Text: {text}')
    print(f'Entities: {entities}')
    print(f'BILUO Tags: {biluo_tags}\n')

# Entrenamiento 
ner = nlp.add_pipe("ner")
for _, annotations in TRAIN_DATA:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

examples = []
for text, annotations in TRAIN_DATA:
    doc = nlp.make_doc(text)
    example = Example.from_dict(doc, annotations)
    examples.append(example)


optimizer = nlp.create_optimizer()
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']

with nlp.disable_pipes(*other_pipes):
    for itn in range(100):
        random.shuffle(examples)
        losses = {}

        
        for batch in spacy.util.minibatch(examples, size=2):
            for example in batch:
                nlp.update([example], drop=0.5, sgd=optimizer, losses=losses)
        print("Losses", losses)

#Modelo NER
output_dir = "ner_architectural_model"
nlp.to_disk(output_dir)
print("Saved model to", output_dir)

#Aplicar el modelo a un conjunto
nlp = spacy.load(output_dir)
doc = nlp(text)
architectural_works = [ent.text for ent in doc.ents if ent.label_ == "ARCH"]
print(architectural_works)
