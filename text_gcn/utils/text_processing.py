import spacy
import gensim.downloader as api
import contextualSpellCheck
from pycontractions import Contractions
import unidecode
import json
import os
import re
# from word2number import w2n


class TextProcessing:
    """
    Class to clean text
    """

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        contextualSpellCheck.add_to_pipe(self.nlp)
        model = api.load("glove-twitter-25")
        self.cont = Contractions(kv_model=model)
        self.cont.load_models()
        dirname = os.path.dirname(__file__)
        with open(os.path.join(dirname, 'acronym.json')) as f:
            self.acronyms = json.load(f)

    def process_text(self, text):
        """
        Processes text as follows:
        1. decode to unicode
        2. lower case the string
        3. expand contractions of english words like ain't
        4. correct spelling mistakes
        5. replace acronyms
        6. replace NE in the text
        Args:
            text: text to be processed
        """
        text = self.unidecode(text)
        text = self.replace_acronyms(text)
        text = self.lower_case(text)
        text = self.expand_contractions(text)
        text = self.correct_spellings(text)
        text = self.replace_named_entity(text)
        return text

    def unidecode(self, text):
        """
        unidecodes the text
        Args:
            text: text to be processed
        """
        return unidecode.unidecode(text)

    def lower_case(self, text):
        """
        lower cases the text
        Args:
            text: text to be processed
        """
        return text.lower()

    def expand_contractions(self, text):
        """
        expands contractions for example, "ain't" expands to "am not"
        Args:
            text: text to be processed
        """
        return list(self.cont.expand_texts([text], precise=True))[0]

    def correct_spellings(self, text):
        """
        corrects spellings from text
        Args:
            text: text to be processed
        """
        doc = self.nlp(text)
        if doc._.performed_spellCheck:
            text = doc._.outcome_spellCheck
        return text

    def replace_acronyms(self, text):
        """
        Replaces acronyms found in English
        For example: ttyl -> talk to you later
        Args:
            text: text to be processed
        """
        for acronym, expansion in self.acronyms.items():
            text = text.replace(acronym.lower(), expansion)
        return text

    def replace_named_entity(self, text):
        """
        Replaces named entity in the text
        For example: $5bn loss estimated in the coming year
                    -> MONEY loss estimated in the coming year
        Args:
            text: text to be processed
        """
        doc = list(self.nlp.pipe([text], disable=[
                   "tagger", "parser", "contextual spellchecker"]))[0]
        for ent in doc.ents:
            text = text.replace(ent.text, ent.label_)
        return text

    def token_list(self, text):
        doc = self.nlp(text)
        tokens = []
        for token in doc:
            tokens += [token.text]
        return tokens

    # def word_to_num(self, text):
        # for token in doc:
        #             if token.pos_ == 'NUM':
        #                 tokens += [w2n.word_to_num(token.lower_)]


if __name__ == "__main__":

    processor = TextProcessing()

    text = "$10m is deep bot's net worth"

    text = processor.process_text(text)

    print(text)
