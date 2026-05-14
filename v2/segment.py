import nltk
import spacy
import torch

from dataclasses import dataclass

TARGET_TAGS = set([
    'VERB',
    'NUM',
    'ADJ',
    'NOUN',
    'PROPN',
])

@dataclass
class Word:
    word: str
    pos: str
    span_start: int
    span_end: int

    def is_content(self):
      return self.pos in TARGET_TAGS

def sentencize(text):
    return nltk.sent_tokenize(text)

def pos_tag(text, nlp):
    # 词性标注
    doc = nlp(text)
    return [(w.text, w.pos_) for w in doc]


def _offsets_overlap(left, right):
    left_start, left_end = left
    right_start, right_end = right
    return left_start < right_end and right_start < left_end

#TODO： 这里对齐可以改成多条encoder并行， 目前是单条encoder逐步对齐
def words_to_token_spans(wpos, tokens, W):
    # Filter out space tokens
    toks_pos = [(t, p) for t, p in wpos if p != "SPACE"]

    # Iterate over words
    i = 0
    cur_word, cur_pos = toks_pos[i]
    
    word_start = 0
    words = []

    for j, subword in enumerate(tokens):
        # if W == subword: continue
    
        if W in subword: # new word
            word_start = j 
    
        # Convert span to string, filter out whitespace
        span = tokens[word_start:j+1]
        # print(span[0], dir(span[0]))
        span = [e.replace(W, "") for e in span]
        cur = ''.join(span)
    
        # equality check
        if cur == cur_word:
    
            # Store span
            w = Word(cur_word, cur_pos, word_start, j+1)
            words.append(w)
    
            # Goto next
            i += 1
            if i >= len(toks_pos): break
            
            cur_word, cur_pos = toks_pos[i]
            word_start = j
    
    
    if not len(words) == len(toks_pos):
      print("Length mismatch")  
      print(words)
      print("-"*30)
      print([t for t,p in toks_pos])

    return words

def align_cot_to_pos(cot_step_text, tokenizer, model_id, nlp):
    del model_id  # Kept in the signature for compatibility with existing callers.

    encoded = tokenizer(
        cot_step_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    indices = torch.tensor(encoded["input_ids"])
    offsets = encoded["offset_mapping"]

    doc = nlp(cot_step_text)
    words = []
    for token in doc:
        if token.pos_ == "SPACE":
            continue
        token_span = (token.idx, token.idx + len(token.text))
        covered_indices = [
            idx
            for idx, offset in enumerate(offsets)
            if offset[0] != offset[1] and _offsets_overlap(offset, token_span)
        ]
        if not covered_indices:
            continue
        words.append(
            Word(
                token.text,
                token.pos_,
                min(covered_indices),
                max(covered_indices) + 1,
            )
        )

    return indices, words
