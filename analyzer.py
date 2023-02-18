import spacy


def analyze(text):
    nlp = spacy.load("custom-stock-pipeline.spacy")
    doc = nlp(text)
    organizations = get_organization(doc.ents)

    return organizations


def get_organization(ents):
    organizations = []

    for entity in ents:
        if entity.label_ == 'COMPANY_SHORTNAME' or entity.label_ == 'COMPANY_NAME':
            organizations.append(entity.text)

    return list(set(organizations))

