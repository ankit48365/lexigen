


def tokenize(text):
    tokens = text.lower().split()
    return tokens

def build_vocab(tokens):
    vocab = list(set(tokens)) # set removes duplicate words and list converts it back to a list
    return vocab

text = "Hello, this is a simple text generator example. This is a simple example."
tokens = tokenize(text)
vocab = build_vocab(tokens)
print("Tokens:", tokens)
print("Vocabulary:", vocab)