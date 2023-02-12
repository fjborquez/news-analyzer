from spacy import load
from pandas import read_csv


def analyze(text):
    nlp = load('en_core_web_sm')
    doc = nlp(text)
    organizations = get_organization(doc.ents)

    return organizations


def get_organization(ents):
    organizations = []

    for entity in ents:
        if entity.label_ == 'ORG' and white_lists_validations(entity.text):
            organizations.append(entity.text)

    return list(set(organizations))


def white_lists_validations(text):
    return csv_validations(text, 'tickers') or csv_validations(text, 'names') \
           or csv_validations(text, 'shortnames')


def csv_validations(text, validation):
    tickers = get_dataset(validation)
    return tickers.count(text) > 0


def get_dataset(name):
    dataset = read_csv('./datasets/' + name + '.csv')
    return dataset['KEYWORD'].tolist()
