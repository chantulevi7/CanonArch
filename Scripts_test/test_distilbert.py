def prepare_ner_training_data(data):
    training_data = []
    
    for sentence, entity_text, entity_label in data:
        sentence = sentence.lower()
        entity_text = entity_text.lower()

        sentence = sentence.replace(",", "")
        sentence = sentence.replace(".", "")
        sentence = sentence.replace(";", "")
        sentence = sentence.replace("!", "")
        sentence = sentence.replace("?", "")

        tokens = sentence.split()
        labels = ["O"] * len(tokens)
        
        # Find the start and end index of the entity text in the tokens
        entity_tokens = entity_text.split()
        for i in range(len(tokens) - len(entity_tokens) + 1):
            if tokens[i:i + len(entity_tokens)] == entity_tokens:
                labels[i] = f"B-{entity_label}"
                for j in range(1, len(entity_tokens)):
                    labels[i + j] = f"I-{entity_label}"
        
        training_data.append({"tokens": tokens, "ner_tags": labels})
    
    return training_data

# Example usage
data = [
    ("Modern environmental culture first manifested itself in Montreal in the work of the Quebecois Art Deco architect-engineer Ernest Cormier, primarily through his Université de Montreal, constructed between 1928 and 1955.", "Université de Montreal", "ARCHITECTURE"),
    ("the Place Ville Marie (1958), an office and shopping complex by I.M. Pei", "Place Ville Marie", "ARCHITECTURE"),
    ("the merchandise mart and rooftop hotel, known as the Place Bonaventure, built to the designs of Ray Affleck in 'corduroy' bush-hammered concrete in 1964.", "Place Bonaventure", "ARCHITECTURE"),
    ("This exhibition was dominated by two monumental works, the first being Buckminster Fuller's US Pavilion, which took the unique form of an enormous geodesic sphere.", "US Pavilion", "ARCHITECTURE"),
    ("The other prominent work, which was also predicated on tetrahedral geometry, was Moshe Safdie's multistorey experimental housing complex, known as Habitat '67", "Habitat '67", "ARCHITECTURE"),
    ("Toronto's entry into modernity began in 1958 with the international competition for Toronto City Hall", "Toronto City Hall", "ARCHITECTURE"),
    ("However, the first move towards a regionally inflected Canadian architecture appeared in Toronto in the work of Ron Thom, whose part-Brutalist, part-late Gothic Revival enclave of Massey College was constructed in the built-up area of the city in 1963", "enclave of Massey College", "ARCHITECTURE"),
    ("This was followed in 1965 by the equally Brutalist, organically planned Scarborough College", "Scarborough College", "ARCHITECTURE"),
    ("On the west coast, the first figure of stature to emerge during the post-war period was Arthur Erickson, whose Simon Fraser University (1962-72), conceived as a landscaped megastructure, was integrated into the rising site of a small mountain near Vancouver.", "Simon Fraser University", "ARCHITECTURE"),
    ("Erickson followed this achievement with two other megastructural works: his bridge-like design for the University of Lethbridge in Alberta (1972), and his Robson Square complex, completed as an axial spine running through the centre of Vancouver in 1986.", "University of Lethbridge", "ARCHITECTURE"),
    ("Erickson followed this achievement with two other megastructural works: his bridge-like design for the University of Lethbridge in Alberta (1972), and his Robson Square complex, completed as an axial spine running through the centre of Vancouver in 1986.", "Robson Square complex", "ARCHITECTURE"),
    ("Even so, the Ecole des Beaux-Arts still exercised an influence, mainly through the activity of French architects who, at the turn of the century, were appointed as professors in a number of leading American universities", "Ecole des Beaux-Arts", "ORG")
]

training_data = prepare_ner_training_data(data)
