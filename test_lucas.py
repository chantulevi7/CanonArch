import spacy
from spacy.training import Example
import random

nlp = spacy.load("en_core_web_lg")

ner = nlp.get_pipe("ner")
ner.add_label("ARCH")

def built_spacy_ner(text, target, type):
    start = str.find(text, target)
    end = start + len(target)

    return (text, {"entities": [(start, end, type)]})

data = [
    ("Modern environmental culture first manifested itself in Montreal in the work of the Quebecois Art Deco architect-engineer Ernest Cormier, primarily through his Université de Montreal, constructed between 1928 and 1955.", "Université de Montreal", "ARCH"),
    ("the Place Ville Marie (1958), an office and shopping complex by I.M. Pei", "Place Ville Marie", "ARCH"),
    ("the merchandise mart and rooftop hotel, known as the Place Bonaventure, built to the designs of Ray Affleck in 'corduroy' bush-hammered concrete in 1964.", "Place Bonaventure", "ARCH"),
    ("This exhibition was dominated by two monumental works, the first being Buckminster Fuller's US Pavilion, which took the unique form of an enormous geodesic sphere.", "US Pavilion", "ARCH"),
    ("The other prominent work, which was also predicated on tetrahedral geometry, was Moshe Safdie's multistorey experimental housing complex, known as Habitat '67", "Habitat '67", "ARCH"),
    ("Toronto's entry into modernity began in 1958 with the international competition for Toronto City Hall", "Toronto City Hall", "ARCH"),
    ("However, the first move towards a regionally inflected Canadian architecture appeared in Toronto in the work of Ron Thom, whose part-Brutalist, part-late Gothic Revival enclave of Massey College was constructed in the built-up area of the city in 1963", "enclave of Massey College", "ARCH"),
    ("This was followed in 1965 by the equally Brutalist, organically planned Scarborough College", "Scarborough College", "ARCH"),
    ("On the west coast, the first figure of stature to emerge during the post-war period was Arthur Erickson, whose Simon Fraser University (1962-72), conceived as a landscaped megastructure, was integrated into the rising site of a small mountain near Vancouver.", "Simon Fraser University", "ARCH"),
    ("Erickson followed this achievement with two other megastructural works: his bridge-like design for the University of Lethbridge in Alberta (1972), and his Robson Square complex, completed as an axial spine running through the centre of Vancouver in 1986.", "University of Lethbridge", "ARCH"),
    ("Erickson followed this achievement with two other megastructural works: his bridge-like design for the University of Lethbridge in Alberta (1972), and his Robson Square complex, completed as an axial spine running through the centre of Vancouver in 1986.", "Robson Square complex", "ARCH"),
    ("Even so, the Ecole des Beaux-Arts still exercised an influence, mainly through the activity of French architects who, at the turn of the century, were appointed as professors in a number of leading American universities", "Ecole des Beaux-Arts", "ORG")
]

TRAIN_DATA = [built_spacy_ner(text, target, type) for text, target, type in data]

# creating an optimizer and selecting a list of pipes NOT to train
optimizer = nlp.create_optimizer()
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']

with nlp.disable_pipes(*other_pipes):
    for itn in range(30):
        random.shuffle(TRAIN_DATA)
        losses = {}

        # batch the examples and iterate over them
        for batch in spacy.util.minibatch(TRAIN_DATA, size=2):
            for text, annotations in batch:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                nlp.update([example], drop=0.35, sgd=optimizer, losses=losses)

print("Final loss: ", losses)

input_text = '''
Two works by Patkau Architects dating from 2016 testify
to their continual inventiveness. The first of these is the
Audain Art Museum, projected as a bridge-like, pitched-roof
gallery spanning over the flood plan of Fitzsimmons Creek in
Whistler, British Columbia. The second is a visitors’ centre
close to the 200-year-old Fort York National Historic Site,
located adjacent to an elevated expressway feeding the
centre of Toronto. Parallel to the monumental autoroute, this
visitors’ centre is clad with inclined plates of corten steel
that house a reception area, a café, an orientation space
and a semi-subterranean ‘time-tunnel’ containing a video
history of the fort.
One of the most significant Ontario practices of the 21st
century is that of Brigitte Shim and Howard Sutcliffe, who
opened their office in Toronto in 1994. Like Patkau Architects
they are acutely aware that their work is located in a vast
northern territory subject to the stresses of a harsh climate.
Shim has characterized the context of their practice in the
following terms:
In Canada, we occupy an enormous landmass with a tiny
population. Whether one lives in the countryside or in the
city, the mythological Canadian landscape permeates our
lives. … The majority of our work is located at the bottom
edge of the Canadian Shield, described as a stone necklace
of ancient metamorphic rock wrapping around Hudson’s
Bay.
2
Their first work on the Canadian Shield was their
Moorelands Camp dining hall of 2002, which took the form
of a large timber shed, built on Lake Kawagama at
Haliburton. It was designed for a long-established summer
camp catering to the needs of urban youth, and for this
reason it was fitted with large, top-hung louvred doors,
enabling the hall to be closed down completely at the end of
the season. Shim-Sutcliffe would parallel this exercise in
modular timber construction in their Harrison Island Camp
[371], built as a vacation house for their own occupation, on
Georgian Bay, Ontario, in 2010. This house was largely
composed of structural insulated panels, which were
shipped to the rocky site by barge and then assembled into
position.
'''
processed_text = ' '.join([line for line in input_text.split('\n') if line != ''])

doc = nlp(processed_text)

for ent in doc.ents:
    print(ent.text, '|', ent.label_)