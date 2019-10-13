# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for using and training tokenizers (char, wordpiece, sentencepiece)"""
from collections import namedtuple
import random
import os
import csv

# import nltk
# nltk.download('punkt')
# from nltk import tokenize as nltk_tokenize
# import sentencepiece as spmpi

from .wordpiece import BertTokenizer, PRETRAINED_VOCAB_ARCHIVE_MAP


class Tokenization(object):
    """
    Tokenization object to hold tokenization, (processed text),and original
    text. Can hold tokenization as Ids or tokens.

    It also holds command tokens (pad, unk, etc.) for the tokenization.
    This allows functions to pad/operate on tokenizations without having
    access to the full tokenizer, just the tokenization.

    Several standard array operations are implemented (insert, append, extend).
    """

    def __init__(self, tokenization, text=None, original_text=None, command_tokens=None, asIds=True):
        self.tokenization = tokenization
        self.text = text
        if self.text is None:
            self.text = self.tokenization
        self.original_text = original_text
        if self.original_text is None:
            self.original_text = self.text
        self.command_tokens = command_tokens
        self.asIds = asIds
        self.parse_command_tokens()

    def set_command_tokens(self, command_tokens):
        self.command_tokens = command_tokens
        return self.parse_command_tokens()

    def parse_command_tokens(self):
        if self.command_tokens is None:
            return
        for command_token in self.command_tokens:
            if self.asIds:
                setattr(self, command_token.name, command_token.Id)
            else:
                setattr(self, command_token.name, command_token.token)

    def __getitem__(self, index):
        return self.tokenization[index]

    def __len__(self):
        return len(self.tokenization)

    def insert(self, idx, other):
        if isinstance(other, (CommandToken, TypeToken)):
            self.tokenization.insert(idx, other.Id)
            if idx == 0:
                self.text.insert(0, other.token)
                self.original_text.insert(0, other.token)
            elif idx == len(self.tokenization) - 1:
                self.text.insert(-1, other.token)
                self.original_text.insert(-1, other.token)
        elif isinstance(other, Tokenization):
            self.tokenization = self.tokenization[:idx] + other.tokenization + self.tokenization[idx:]
        else:
            self.tokenization = self.tokenization[:idx] + other.tokenization + self.tokenization[idx:]

    def append(self, other):
        if isinstance(other, (CommandToken, TypeToken)):
            self.tokenization.append(other.Id)
            self.text.append(other.token)
            self.original_text.append(other.token)
        elif isinstance(other, Tokenization):
            self.tokenization.extend(other.tokenization)
            self.text += other.text
            self.original_text += other.original_text
        else:
            self.tokenization.append(other)
        return self

    def extend(self, other):
        if isinstance(other, (CommandToken, TypeToken)):
            self.tokenization.append(other.Id)
            self.text.append(other.token)
            self.original_text.append(other.token)
        elif isinstance(other, list) and isinstance(other[0], (CommandToken, TypeToken)):
            self.tokenization.extend([o.Id for o in other])
            self.text += [o.token for o in other]
            self.original_text += [o.token for o in other]
        elif isinstance(other, Tokenization):
            self.tokenization.extend(other.tokenization)
            self.text += other.text
            self.original_text += other.original_text
        else:
            self.tokenization.extend(other)
        return self


"""define some default command tokens for the tokenizer to use"""
token_format = "<{0}>"

COMMAND_TUPLE = namedtuple('CommandToken', ('name', 'token', 'Id'))


def prep_command_tokens(tokenlist, token_format=token_format):
    return [CommandToken(tok[0], token_format.format(tok[0]), tok[1]) for tok in tokenlist]


class CommandToken(object):
    def __init__(self, name, token, Id):
        self.name = name
        self.token = token
        self.Id = Id

    def __str__(self):
        return str(COMMAND_TUPLE(self.name, self.token, self.Id))


DEFAULT_COMMAND_TOKENS = [
    ('pad', 0),
    ('eos', 1),
    ('bos', 2),
    ('unk', 3),
    ('sep', 4),
    ('L2R', 5),
    ('ENC', 6),
    ('MASK', 7),
]
DEFAULT_COMMAND_TOKENS = prep_command_tokens(DEFAULT_COMMAND_TOKENS)

"""define some default type tokens for bert training"""

TYPE_TUPLE = namedtuple('TypeToken', ('name', 'token', 'Id'))


def prep_type_tokens(tokenlist, token_format=token_format):
    return [TypeToken(tok[0], token_format.format(tok[0]), tok[1]) for tok in tokenlist]


class TypeToken(object):
    def __init__(self, name, token, Id):
        self.name = name
        self.token = token
        self.Id = Id

    def __str__(self):
        return str(TYPE_TUPLE(self.name, self.token, self.Id))


DEFAULT_TYPE_TOKENS = [
    ('function', 0),
    ('command', 1),
    ('str0', 2),
    ('str1', 3),
    ('str2', 4),
    ('embedding0', 5),
    ('embedding1', 6),
    ('embedding2', 7),
    ('arg0', 8),
    ('arg1', 9),
    ('arg2', 10),
]
DEFAULT_TYPE_TOKENS = prep_type_tokens(DEFAULT_TYPE_TOKENS)


class Tokenizer(object):
    """
    Tokenizer object that handles text tokenization, command tokens, and type tokens.

    Command tokens and text tokens are stored together in one mapping of size
    `len(text_tokenizer)+len(command_tokens)`. Command tokens are stored as first
    `len(command_tokens)` tokens. Token idx is stored at `idx+len(command_tokens)`.

    Token types are stored in a separate mapping of size `len(type_tokens)`.
    """

    def __init__(self, text_tokenizer, command_tokens=None, type_tokens=None):
        # set text tokenizer
        self.text_tokenizer = text_tokenizer
        if not hasattr(self, 'num_text_tokens'):
            self.num_text_tokens = len(self.text_tokenizer)

        # set command tokens
        if command_tokens is None:
            command_tokens = DEFAULT_COMMAND_TOKENS
        self._command_tokens = command_tokens
        self.command_name_map = {tok.name: tok for tok in self._command_tokens}
        self.command_token_map = {tok.token: tok for tok in self._command_tokens}
        self.command_id_map = {tok.Id: tok for tok in self._command_tokens}
        if not hasattr(self, 'num_command_tokens'):
            self.num_command_tokens = len(self._command_tokens)
        if not hasattr(self, 'num_tokens'):
            self.num_tokens = self.num_command_tokens + self.num_text_tokens

        # set type tokens
        if type_tokens is None:
            type_tokens = DEFAULT_TYPE_TOKENS
        self.type_tokens = type_tokens
        self.type_name_map = {tok.name: tok for tok in self.type_tokens}
        self.type_token_map = {tok.token: tok for tok in self.type_tokens}
        self.type_id_map = {tok.Id: tok for tok in self.type_tokens}
        if not hasattr(self, 'num_type_tokens'):
            self.num_type_tokens = len(self.type_tokens)

        # parse tokens and vocabs from tokenizer
        self._tokens = list(self.command_token_map.keys()) + list(self.text_tokenizer.tokens)
        self._vocab = {t: Id for Id, t in self.command_id_map.items()}
        self._vocab.update({t: Id + self.num_command_tokens for t, Id in self.text_tokenizer.vocab.items()})

        self._text_tokens = list(self.text_tokenizer.tokens)
        self._text_token_vocab = {t: Id + self.num_command_tokens for t, Id in self.text_tokenizer.vocab.items()}

        self._command_token_tokens = list(self.command_token_map.keys())
        self._command_token_vocab = {t: Id for Id, t in self.command_id_map.items()}

        self._token_types = list(self.type_token_map.keys())
        self._token_type_vocab = {t: Id for Id, t in self.type_id_map.items()}

    def __call__(self, text, process_fn=None):
        """run preprocessing and encode text as Ids"""
        return self.EncodeAsIds(text, process_fn=process_fn)

    def __len__(self):
        """total number of tokens"""
        return self.num_tokens

    def get_command(self, name):
        """get command token corresponding to `name`"""
        return self.command_name_map[name]

    def get_type(self, name):
        """get type token corresponding to `name`"""
        return self.type_name_map[name]

    @property
    def tokens(self):
        """list (or iterable) of all tokens for tokenizer"""
        return self._tokens

    @property
    def vocab(self):
        """dictionary mapping tokens to ids for tokenizer"""
        return self._vocab

    @property
    def token_types(self):
        """list (or iterable) of all token types for tokenizer"""
        return self._token_types

    @property
    def token_type_vocab(self):
        """dictionary mapping token types to ids for tokenizer"""
        return self._token_type_vocab

    @property
    def command_tokens(self):
        """list (or iterable) of all command tokens for tokenizer"""
        return self._command_token_tokens

    @property
    def command_token_vocab(self):
        """dictionary mapping command tokens to ids for tokenizer"""
        return self._command_token_vocab

    @property
    def text_tokens(self):
        """list (or iterable) of text tokens for text tokenizer"""
        return self._text_tokens

    @property
    def text_token_vocab(self):
        """dictionary mapping text tokens to ids for text tokenizer"""
        return self._text_token_vocab

    def EncodeAsIds(self, text, process_fn=None):
        """
        encode text using text tokenizer and shift Id values for command tokens
        """
        tokenization = self.text_tokenizer.EncodeAsIds(text, process_fn=process_fn)
        tokenization.tokenization = [t + self.num_command_tokens for t in tokenization.tokenization]
        tokenization.set_command_tokens(self._command_tokens)
        return tokenization

    def EncodeAsTokens(self, text, process_fn=None):
        """
        encode text as tokens using text tokenizer
        """
        tokenization = self.text_tokenizer.EncodeAsTokens(text, process_fn=process_fn)
        tokenization.set_command_tokens(self._command_tokens)
        return tokenization

    def IdToToken(self, Id, type_token=False):
        """convert Id to token accounting for command and type tokens"""
        if isinstance(Id, (TypeToken, CommandToken)):
            return Id.token
        if type_token:
            return self.type_id_map[Id].token
        if Id < self.num_command_tokens:
            return self.command_id_map[Id].token
        return self.text_tokenizer.IdToToken(Id - self.num_command_tokens)

    def TokenToId(self, token, type_token=False):
        """convert token to Id accounting for command and type tokens"""
        if isinstance(token, (TypeToken, CommandToken)):
            return token.Id
        if type_token:
            return self.type_token_map[token].Id
        if token in self.command_token_map:
            return self.command_token_map[token].Id
        return self.text_tokenizer.TokenToId(token) + self.num_command_tokens

    def DecodeIds(self, Ids, type_token=False):
        """
        convert Ids to tokens accounting for command and type tokens, tokens
        are joined and returned as a string.
        """
        if type_token:
            return ' '.join(Id.token if isinstance(Id, TypeToken) else self.type_id_map[Id].token for Id in Ids)
        rtn_strs = []
        current_str = []
        if isinstance(Ids, Tokenization):
            Ids = Ids.tokenization
        for Id in Ids:
            if isinstance(Id, CommandToken):
                rtn_strs.append(self.text_tokenizer.DecodeIds(current_str))
                current_str = []
                rtn_strs.append(t.token)
            elif Id < self.num_command_tokens:
                rtn_strs.append(self.text_tokenizer.DecodeIds(current_str))
                current_str = []
                rtn_strs.append(self.command_id_map[Id].token)
            else:
                current_str.append(Id - self.num_command_tokens)
        if current_str != []:
            rtn_strs.append(self.text_tokenizer.DecodeIds(current_str))
        return ' '.join(rtn_strs)

    def DecodeTokens(self, Tokens, type_token=False):
        """
        convert tokens to a string accounting for command and type tokens.
        """
        if type_token:
            return ' '.join(t.token if isinstance(t, TypeToken) else t for t in Tokens)
        rtn_strs = []
        current_str = []
        if isinstance(Tokens, Tokenization):
            Tokens = Tokens.tokenization
        for t in Tokens:
            if isinstance(t, CommandToken):
                rtn_strs.append(self.text_tokenizer.DecodeTokens(current_str))
                current_str = []
                rtn_strs.append(t.token)
            elif t in self.command_token_map:
                rtn_strs.append(self.text_tokenizer.DecodeTokens(current_str))
                current_str = []
                rtn_strs.append(t)
            else:
                current_str.append(t)
        if current_str != []:
            rtn_strs.append(self.text_tokenizer.DecodeTokens(current_str))
        return ' '.join(rtn_strs)


class TextTokenizer(object):
    """
    Interface for text tokenizer
    """

    def __init__(self):
        if not hasattr(self, 'num_text_tokens'):
            self.num_text_tokens = 0
        if not hasattr(self, 'num_tokens'):
            self.num_tokens = self.num_text_tokens

    def __call__(self, text, process_fn=None):
        return self.EncodeAsIds(text, process_fn)

    def __len__(self):
        return self.num_text_tokens

    @property
    def tokens(self):
        """list (or iterable) of text tokens for text tokenizer"""
        raise NotImplementedError('TextTokenizer tokens property not implemented')

    @property
    def vocab(self):
        """dictionary mapping tokens to ids"""
        raise NotImplementedError('TextTokenizer vocab property not implemented')

    @staticmethod
    def exists(model_path):
        """check if the filepath for a text tokenizer exists"""
        raise NotImplementedError('TextTokenizer exists method not implemented')

    def Train(self, corpus):
        """train a tokenizer on a data corpus and save model for future use"""
        raise NotImplementedError('TextTokenizer Train not implemented')

    def EncodeAsIds(self, text, process_fn=None):
        """
        Preprocess text and encode as ids. Return a tokenization object with
        original text, processed text, and id tokenization.
        """
        raise NotImplementedError('TextTokenizer EncodeAsIds not implemented')

    def EncodeAsTokens(self, text, process_fn=None):
        """
        Preprocess text and encode as tokens. Return a tokenization object with
        original text, processed text, and token tokenization.
        """
        raise NotImplementedError('TextTokenizer EncodeAsTokens not implemented')

    def IdToToken(self, Id):
        """Convert an Id to Token. Reverse lookup of self.vocab"""
        raise NotImplementedError('TextTokenizer IdToToken not implemented')

    def TokenToId(self, token):
        """Convert a Token to Id. Lookup of self.vocab"""
        raise NotImplementedError('TextTokenizer TokenToId not implemented')

    def DecodeIds(self, Ids):
        """Convert a list or tokenization object of Ids to a text string"""
        raise NotImplementedError('TextTokenizer DecodeIds not implemented')

    def DecodeTokens(self, Tokens):
        """Convert a list or tokenization object of tokens to a text string"""
        raise NotImplementedError('TextTokenizer DecodeTokens not implemented')


class CharacterLevelTokenizer(TextTokenizer):
    """
    Text tokenizer for ASCII-256 Character Level Tokenization.
    """

    def __init__(self, **kwargs):
        self.num_text_tokens = 256
        super(CharacterLevelTokenizer, self).__init__()
        self._tokens = [self.IdToToken(Id) for Id in range(self.num_text_tokens)]
        self._vocab = {t: i for i, t in enumerate(self._tokens)}

    def __len__(self):
        return 256

    @staticmethod
    def exists(model_path):
        return True

    def Train(self, corpus):
        pass

    @property
    def tokens(self):
        return self._tokens

    @property
    def vocab(self):
        return self._vocab

    def EncodeAsIds(self, text, process_fn=None):
        """convert text to ascii 256 Ids"""
        processed_text = text
        if process_fn is not None:
            processed_text = process_fn(processed_text)
            processed_text = str(processed_text)
        tokens = [self.TokenToId(c) for c in processed_text]
        return Tokenization(tokens, processed_text, text)

    def EncodeAsTokens(self, text, process_fn=None):
        """convert text to ascii 256 characters"""
        processed_text = text
        if process_fn is not None:
            processed_text = process_fn(processed_text)
        processed_text = str(processed_text)
        tokens = [c for c in processed_text]
        return Tokenization(tokens, processed_text, text, asIds=False)

    def IdToToken(self, Id):
        """ascii index to character"""
        return chr(Id)

    def TokenToId(self, token):
        """ascii character to index"""
        return ord(token)

    def DecodeIds(self, Ids):
        """converts ascii ids to tokens before joining them into text"""
        if isinstance(Ids, Tokenization):
            Ids = Ids.tokenization
        return ''.join([self.IdToToken(tok) for tok in Ids])

    def DecodeTokens(self, Tokens):
        """just concatenates ascii tokens into text"""
        if isinstance(Tokens, Tokenization):
            Tokens = Tokens.tokenization
        return ''.join(Tokens)


class BertWordPieceTokenizer(Tokenizer):
    """
    Loads a pretrained WordPiece tokenizer from `cache_dir` for tokenization
    in BERT training. Default to bert-large-uncased tokenizer.
    """

    def __init__(self, tokenizer_model_type=None, cache_dir=None, **kwargs):
        # default to bert-large-uncased tokenizer
        if tokenizer_model_type not in PRETRAINED_VOCAB_ARCHIVE_MAP:
            tokenizer_model_type = 'bert-large-uncased'
        print('loading BertWordPieceTokenizer (', tokenizer_model_type, ') from cache_dir ', cache_dir)
        do_lower_case = not ('-cased' in tokenizer_model_type or 'chinese' in tokenizer_model_type)
        self.text_tokenizer = BertTokenizer.from_pretrained(tokenizer_model_type, do_lower_case=do_lower_case, cache_dir=cache_dir)
        print('loaded', tokenizer_model_type)
        # disable max len warnings by increasing max len
        self.text_tokenizer.max_len = int(1e12)

        # set command tokens from wordpiece tokenizer values
        self.num_command_tokens = 5
        self.num_tokens = len(self.text_tokenizer.vocab)
        self.num_text_tokens = self.num_tokens - 5
        self.num_type_tokens = 2

        self._command_tokens = [
            CommandToken('pad', '[PAD]', self.text_tokenizer.vocab['[PAD]']),
            CommandToken('ENC', '[CLS]', self.text_tokenizer.vocab['[CLS]']),
            CommandToken('MASK', '[MASK]', self.text_tokenizer.vocab['[MASK]']),
            CommandToken('unk', '[UNK]', self.text_tokenizer.vocab['[UNK]']),
            CommandToken('sep', '[SEP]', self.text_tokenizer.vocab['[SEP]']),
            CommandToken('sep_1', '[SEP_1]', self.text_tokenizer.vocab['[unused10]']),
            CommandToken('style0', '[STYLE0]', self.text_tokenizer.vocab['[unused20]']),
            CommandToken('style1', '[STYLE1]', self.text_tokenizer.vocab['[unused21]']),
            CommandToken('cls0', '[CLS0]', self.text_tokenizer.vocab['[CLS]']),
            CommandToken('cls1', '[CLS1]', self.text_tokenizer.vocab['[unused31]']),
            CommandToken('cls2', '[CLS2]', self.text_tokenizer.vocab['[unused32]']),
            CommandToken('cls3', '[CLS3]', self.text_tokenizer.vocab['[unused33]']),
            CommandToken('cls4', '[CLS4]', self.text_tokenizer.vocab['[unused34]']),
        ]
        self.command_name_map = {tok.name: tok for tok in self._command_tokens}
        self.command_token_map = {tok.token: tok for tok in self._command_tokens}
        self.command_id_map = {tok.Id: tok for tok in self._command_tokens}

        # set type tokens
        self.type_tokens = [
            TypeToken('str0', '<str0>', 0),
            TypeToken('str1', '<str1>', 1),
        ]
        self.type_name_map = {tok.name: tok for tok in self.type_tokens}
        self.type_token_map = {tok.token: tok for tok in self.type_tokens}
        self.type_id_map = {tok.Id: tok for tok in self.type_tokens}

        # parse tokens and vocabs from tokenizer

        self._tokens = list(self.text_tokenizer.vocab.keys())
        self._vocab = {k: v for k, v in self.text_tokenizer.vocab.items()}

        self._text_tokens = list(self._tokens)
        self._text_token_vocab = {k: v for k, v in self.text_tokenizer.vocab.items()}

        self._command_token_tokens = list(self.command_token_map.keys())
        self._command_token_vocab = {t: Id for Id, t in self.command_id_map.items()}

        self._token_types = list(self.type_token_map.keys())
        self._token_type_vocab = {t: Id for Id, t in self.type_id_map.items()}

    def pad(self):
        return self.get_command("pad").Id

    def cls(self):
        return self.get_command("ENC").Id

    def eos(self):
        return self.sep()

    def mask(self):
        return self.get_command("MASK").Id

    def sep(self):
        return self.get_command("sep").Id

    def sep1(self):
        return self.get_command("sep_1").Id

    def unk(self):
        return self.get_command("unk").Id

    def style(self, stype=0):
        name = "style{}".format(stype)
        return self.get_command(name).Id

    def cls_style(self, style=0):
        name = "cls{}".format(style)
        return self.get_command(name).Id

    def EncodeAsIds(self, text, process_fn=None):
        """convert text to wordpiece Ids"""
        processed_text = text
        if process_fn is not None:
            processed_text = process_fn(processed_text)
        tokens = self.text_tokenizer.tokenize(processed_text)
        Ids = self.text_tokenizer.convert_tokens_to_ids(tokens)
        return Tokenization(Ids, processed_text, text)

    def convert_text_to_ids(self, text, process_fn=None):
        tokens = self.EncodeAsIds(text, process_fn=process_fn)
        return tokens.tokenization

    def convert_ids_to_text(self, ids):
        return self.DecodeIds(ids)

    def EncodeAsTokens(self, text, process_fn=None):
        """convert wordpiece token to Id"""
        processed_text = text
        if process_fn is not None:
            processed_text = process_fn(processed_text)
        tokens = self.text_tokenizer.tokenize(processed_text)
        return Tokenization(tokens, processed_text, text, asIds=False)

    def IdToToken(self, Id, type_token=False):
        """convert Id to sentencpiece token"""
        if isinstance(Id, (TypeToken, CommandToken)):
            return Id.token
        if type_token:
            return self.type_id_map[Id].token
        return self.text_tokenizer.ids_to_tokens[Id]

    def TokenToId(self, token, type_token=False):
        """convert sentencpiece token to Id"""
        if isinstance(token, (TypeToken, CommandToken)):
            return token.Id
        if type_token:
            return self.type_token_map[token].Id
        return self.text_tokenizer.vocab[token]

    def DecodeIds(self, Ids, type_token=False):
        """converts ids to wordpiece tokens and joins them as a text string"""
        if type_token:
            return ' '.join(Id.token if isinstance(Id, TypeToken) else self.type_id_map[Id].token for Id in Ids)
        if isinstance(Ids, Tokenization):
            Ids = Ids.tokenization
        Tokens = []
        for Id in Ids:
            Tokens.append(self.text_tokenizer.ids_to_tokens[Id] if Id != -1 else '-1')
        Tokens = self.text_tokenizer.convert_ids_to_tokens(Ids)
        return ' '.join(Tokens)

    def DecodeTokens(self, Tokens, type_token=False):
        """converts wordpiece tokens to a text string"""
        if type_token:
            return ' '.join(t.token if isinstance(t, TypeToken) else t for t in Tokens)
        if isinstance(Tokens, Tokenization):
            Tokens = Tokens.tokenization
        return ' '.join(Tokens)
