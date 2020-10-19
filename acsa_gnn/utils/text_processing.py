import os
import re
import json
import spacy
import unidecode
import contextualSpellCheck
import gensim.downloader as api
from pycontractions import Contractions

from config import configuration as cfg


class TextProcessing:
    """
    Class to clean text
    """

    def __init__(self, nlp=spacy.load("en_core_web_sm")):
        self.nlp = nlp
        contextualSpellCheck.add_to_pipe(self.nlp)
        model = api.load(cfg['embeddings']['embedding_file'])
        self.cont = Contractions(kv_model=model)
        self.cont.load_models()
        dirname = os.path.dirname(__file__)
        with open(os.path.join(dirname, 'acronym.json')) as f:
            self.acronyms = json.load(f)

    def process_text(self, text):
        """
        Processes text as follows:
        1. decode to unicode
        2. remove extra repeated special characters
        3. put space around the special characters
        4. Remove extra whitespaces
        5. replace acronyms
        6. expand contractions of english words like ain't
        7. correct spelling mistakes
        8. replace NE in the text
        9. lower case the string
        Args:
            text: text to be processed
        """
        text = self.unidecode(text)
        text = self.remove_repeated_chars(text)
        text = self.put_space_around_special_chars(text)
        text = self.remove_extra_whitespaces(text)
        text = self.replace_acronyms(text)
        text = self.expand_contractions(text)
        text = self.correct_spellings(text)
        text = self.replace_named_entity(text)
        text = self.lower_case(text)
        return text

    def remove_repeated_chars(self, text):
        """
        Removes repeated instances of consecutive special chars
        Args:
            text: text to be processed
        """
        text = re.sub(r'([!@#$%^&*,./?\'";:\\])\1+', r'\1', text)
        return text

    def put_space_around_special_chars(self, text):
        """
        Puts space around special chars like '[({$&*#@!'
        Args:
            text: text to be processed
        """

        chars = ['$', '?', '%', '@', '!', '#', '^', '*', '&', '"',
                 ':', ';', '/', '\\', ',', '+',
                 '(', ')', '[', ']', '{', '}', '<', '>']

        for char in chars:
            text = text.replace(char, ' '+char+' ')
        return text

    def remove_extra_whitespaces(self, text):
        """
        Removes extra whitespaces from the text
        Args:
            text: text to be processed
        """
        return text.strip()

    def unidecode(self, text):
        """
        unidecodes the text
        Args:
            text: text to be processed
        """
        return unidecode.unidecode(text.lower())

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
        return list(self.cont.expand_texts([text.lower()], precise=True))[0]

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
            text = text.replace(' '+acronym.lower()+' ', ' '+expansion.lower()+' ')
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
