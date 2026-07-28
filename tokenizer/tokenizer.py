class Tokenizer:
    def __init__(self,vocab):
        self.vocab = vocab
        self.inverse_vocab = {v:k for k,v in vocab.items()}

    def encode(self, text):
        words = text.replace("."," .").split()
        ids = [self.vocab[word] for word in words]
        return ids

    def decode(self,ids):
        words = [self.inverse_vocab[id] for id in ids]
        text = " ".join(words).replace(" .", ".")
        return text




