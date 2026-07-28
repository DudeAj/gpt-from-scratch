from collections import Counter
import json

from configs.config import VOCAB_SIZE

class BPETokenizer:

    def __init__(self):
        self.vocab = {}
        self.merges = []
        self.stoi = {}
        self.itos = {}

    def train(self, text, vocab_size):
        """
        Train a BPE tokenizer.

        Parameters
        ----------
        text : str
            Training corpus.

        vocab_size : int
            Target vocabulary size.
        """
        self.vocab = self._build_vocab(text)
        symbols = set()

        for word in self.vocab:
            symbols.update(word)

        while len(symbols)< vocab_size:
            best_pair = self._get_best_pair()

            if best_pair is None:
                break

            self._merge_pair(best_pair)

            symbols = set()

            for word in self.vocab:
                symbols.update(word)
        self._build_token_mappings()



    def encode(self, text):
        """
        Convert text into token IDs.

        Unknown characters (not seen during training) are silently skipped
        rather than raising a KeyError, which would crash generation on any
        punctuation or casing the tokenizer hasn't seen.
        """
        token_ids = []

        for word in text.split():
            symbols = list(word) + ["</w>"]

            for pair in self.merges:
                symbols = self._merge_symbols(symbols, pair)

            for symbol in symbols:
                if symbol in self.stoi:
                    token_ids.append(self.stoi[symbol])
                # silently skip unknown symbols

        return token_ids

    def decode(self, ids):
        """
        Convert token IDs back into text.

        After BPE training, tokens are often fully-merged strings that
        already contain </w> — e.g. "the</w>", "and</w>", "h".
        We can't check `symbol == "</w>"` because the marker is usually
        embedded inside a larger token, not emitted as its own token.

        The correct approach:
          - Strip </w> from the end of every symbol — that marks a word boundary
          - Accumulate sub-word fragments into the current word
          - Flush the word to the list whenever we hit a </w>-terminated token
        """
        words        = []
        current_word = ""

        for idx in ids:
            symbol = self.itos.get(idx, "")

            if symbol.endswith("</w>"):
                # This token completes a word — strip the marker and flush
                current_word += symbol[:-len("</w>")]
                if current_word:
                    words.append(current_word)
                current_word = ""
            else:
                # Sub-word fragment — keep accumulating
                current_word += symbol

        # Flush any trailing fragment (incomplete word at end of sequence)
        if current_word:
            words.append(current_word)

        return " ".join(words)

    def save(self, path):
        """
        Save the trained tokenizer to disk.
        """

        tokenizer_data = {

            "merges": self.merges,

            "stoi": self.stoi,

            "itos": self.itos
        }

        with open(path, "w", encoding="utf-8") as f:

            json.dump(tokenizer_data, f, indent=4)

    def load(self, path):
        """
        Load a trained tokenizer from disk.
        """

        with open(path, "r", encoding="utf-8") as f:
            tokenizer_data = json.load(f)

        # Restore merge rules
        self.merges = [
            tuple(pair)
            for pair in tokenizer_data["merges"]
        ]

        # Restore symbol -> id mapping
        self.stoi = tokenizer_data["stoi"]

        # Restore id -> symbol mapping
        self.itos = {
            int(idx): symbol
            for idx, symbol in tokenizer_data["itos"].items()
        }

    def _build_vocab(self, text):
        """
        Build the initial BPE vocabulary.

        Parameters
        ----------
        text : str
            Raw training corpus.

        Returns
        -------
        dict
            {
                tuple(symbols): frequency
            }
        """

        vocab = Counter()

        words = text.split()
        for word in words:
            symbols = tuple(list(word)+["</w>"])
            vocab[symbols] +=1
        return dict(vocab)

    def _get_pair_counts(self):
        """
        Count frequencies of all adjacent symbol pairs.

        Returns
        -------
        Counter
            {
                (symbol1, symbol2): frequency
            }
        """
        pair_counts = Counter()

        for word, frequency in self.vocab.items():
            for i in range(len(word)-1):
                pair = (word[i],word[i+1])
                pair_counts[pair] += frequency
        return pair_counts

    def _get_best_pair(self):
        """
        Find the most frequent adjacent symbol pair.

        Returns
        -------
        tuple[str, str] | None
            The pair with the highest frequency,
            or None if no pairs exist.
        """

        pair_counts = self._get_pair_counts()

        if not pair_counts:
            return None

        best_pair = max(pair_counts, key=pair_counts.get)

        return best_pair

    def _merge_pair(self, pair):
        """
        Merge on

        """
        merged_symbol = "".join(pair)

        new_vocab = {}

        for word, frequency in self.vocab.items():
            new_word = []
            i = 0
            while i <len(word):
                if (i<len(word)-1 and word[i] == pair[0] and word[i+1] == pair[1]):
                    new_word.append(merged_symbol)
                    i += 2

                else:
                    new_word.append(word[i])
                    i += 1
            new_vocab[tuple(new_word)] = frequency
        self.vocab = new_vocab
        self.merges.append(pair)

    def _build_token_mappings(self):
        """
        Build token <-> ID mappings after BPE training.

        Creates:
            self.stoi : {symbol -> token_id}
            self.itos : {token_id -> symbol}
        """

        # Collect all unique symbols
        symbols = set()

        for word in self.vocab:
            symbols.update(word)

        # Sort to ensure deterministic token IDs
        symbols = sorted(symbols)

        # String -> Integer
        self.stoi = {
            symbol: idx
            for idx, symbol in enumerate(symbols)
        }

        # Integer -> String
        self.itos = {
            idx: symbol
            for symbol, idx in self.stoi.items()
        }

    def _merge_symbols(self, symbols, pair):
        """
        Merge a single pair inside one list of symbols.

        Parameters
        ----------
        symbols : list[str]
            Symbol sequence for one word.

        pair : tuple[str, str]
            Pair to merge.

        Returns
        -------
        list[str]
            Updated symbol sequence.
        """

        merged = []
        merged_symbol = "".join(pair)

        i = 0

        while i < len(symbols):

            if (
                i < len(symbols) - 1
                and symbols[i] == pair[0]
                and symbols[i + 1] == pair[1]
            ):

                merged.append(merged_symbol)
                i += 2

            else:

                merged.append(symbols[i])
                i += 1

        return merged