import spacy
from pandas import read_csv
from spacy.tokens import DocBin
from tqdm import tqdm


def convert():
    df = read_csv("datasets/labeled_data.csv")
    l1 = []
    l2 = []

    for i in range(0, len(df["KEYWORD"])):
        l1.append(df["KEYWORD"][i])
        l2.append({"entities": [(0, len(df["KEYWORD"][i]), df["LABEL"][i])]})

    to_train_data = list(zip(l1,l2))
    create_bin(to_train_data)


def create_bin(to_train_data):
    db = DocBin()
    nlp = spacy.blank("en")

    for text, annot in tqdm(to_train_data):
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in annot["entities"]:
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                print("Skipping entity")
            else:
                ents.append(span)
        doc.ents = ents
        db.add(doc)

    db.to_disk("./labeled_trained_data.spacy")


def main():
    convert()


main()
