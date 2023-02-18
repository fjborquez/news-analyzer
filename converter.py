import spacy
from pandas import read_csv
from spacy.tokens import DocBin
from tqdm import tqdm


def convert():
    df = read_csv("datasets/labeled_data.csv")
    nlp = spacy.load("en_core_web_lg")
    ruler = nlp.add_pipe("entity_ruler", before="ner", config={"phrase_matcher_attr": "LOWER"})

    for i in range(0, len(df["KEYWORD"])):
        keyword = df["KEYWORD"][i]
        label = df["LABEL"][i]

        ruler.add_patterns([{
            "label": label, "pattern": keyword
        }])

    nlp.to_disk("custom-stock-pipeline.spacy")

def main():
    convert()


main()
