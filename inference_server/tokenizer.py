"""Character-level vocabulary construction and prompt encoding/decoding."""


def build_vocab(corpus, special_tokens):
    """Build character-level id tables from a corpus, with special tokens taking the lowest ids."""
    unique_characters = sorted({character for text in corpus for character in text})
    id_to_token = list(special_tokens) + unique_characters
    token_to_id = {token: index for index, token in enumerate(id_to_token)}

    return {'token_to_id': token_to_id, 'id_to_token': id_to_token}


def encode_prompt(text, vocab, add_bos=True):
    """Encode text into token ids, optionally prefixed with ``<bos>``; unknowns map to ``<unk>``."""
    token_to_id = vocab['token_to_id']
    has_unk = '<unk>' in token_to_id

    token_ids = []
    if add_bos:
        token_ids.append(token_to_id['<bos>'])

    for character in text:
        if character in token_to_id:
            token_ids.append(token_to_id[character])
        elif has_unk:
            token_ids.append(token_to_id['<unk>'])

    return token_ids


def decode_tokens(token_ids, vocab, skip_special=True):
    """Join token ids back into text, optionally dropping ``<...>`` special tokens."""
    id_to_token = vocab['id_to_token']

    pieces = []
    for token_id in token_ids:
        token = id_to_token[token_id]
        is_special = token.startswith('<') and token.endswith('>')
        if is_special and skip_special:
            continue
        pieces.append(token)

    return ''.join(pieces)
