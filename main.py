import nltk
sentence = """At eight o'clock on Thursday morning
... Arthur didn't feel very good."""
tokens = nltk.word_tokenize(sentence)
tokens
tagged = nltk.pos_tag(tokens)
tagged[0:6]
entities = nltk.chunk.ne_chunk(tagged)
# print(entities)
from nltk.corpus import treebank
t = treebank.parsed_sents('wsj_0001.mrg')[0]
# t.draw()
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
s = '''Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\nThanks.'''
d = TreebankWordDetokenizer()
t = TreebankWordTokenizer()
toks = t.tokenize(s)
d.detokenize(toks)
expected_tokens = ['Good', 'muffins', 'cost', '$', '3.88', 'in', 'New', '-LRB-', 'York', '-RRB-', '.', 'Please', '-LRB-', 'buy', '-RRB-', 'me', 'two', 'of', 'them.', '-LRB-', 'Thanks', '-RRB-', '.']
expected_tokens == t.tokenize(s, convert_parentheses=True)
expected_detoken = 'Good muffins cost $3.88 in New (York). Please (buy) me two of them. (Thanks).'
expected_detoken == d.detokenize(t.tokenize(s, convert_parentheses=True), convert_parentheses=True)
from nltk.tokenize.treebank import TreebankWordDetokenizer
toks = ['hello', ',', 'i', 'ca', "n't", 'feel', 'my', 'feet', '!', 'Help', '!', '!']
twd = TreebankWordDetokenizer()
twd.detokenize(toks)
toks = ['hello', ',', 'i', "can't", 'feel', ';', 'my', 'feet', '!', 'Help', '!', '!', 'He', 'said', ':', 'Help', ',', 'help', '?', '!']
twd.detokenize(toks)
from nltk.tokenize import TreebankWordTokenizer
s = '''Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\nThanks.'''
TreebankWordTokenizer().tokenize(s)

s = "They'll save and invest more."
TreebankWordTokenizer().tokenize(s)

s = "hi, my name can't hello,"
TreebankWordTokenizer().tokenize(s)

