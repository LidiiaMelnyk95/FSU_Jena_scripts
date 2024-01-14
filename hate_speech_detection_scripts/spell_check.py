import pandas as pd
from symspellpy import SymSpell, Verbosity
from config import config

class SymSpellChecker():
    def __init__(self):
        self.sym_spell = SymSpell()
        self.corpus_path = config['russian_corpus_path']
        self.symspell_dictionary = self.sym_spell.load_dictionary(self.corpus_path, term_index=0, count_index=1, separator=None, encoding='utf-8-sig')

    def russian_spell_check(self, text):
        edited_text = self.sym_spell.lookup(text, Verbosity.CLOSEST, max_edit_distance=0, include_unknown=True, transfer_casing=False, ignore_token=r"\w+\d")
        return edited_text

def main():
    spell_checker = SymSpellChecker()
    df = pd.read_csv(config['comments_path'])
    df = df[df['predicted_language'] == 'Russian']
    df['edited'] = df['edited'].astype(str).apply(lambda x: spell_checker.russian_spell_check(x))
    df.to_csv(config['russian_output_path'], sep=',')

if __name__ == '__main__':
    main()
