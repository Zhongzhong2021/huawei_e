"""Distinguish a present text placeholder from an available lexical token."""
import numpy as np

POLICIES = ('retain', 'mask_unk')

def text_state(tokens, valid, present, policy='retain'):
    if policy not in POLICIES:
        raise ValueError(f'Unknown text observation policy: {policy}')
    present = np.asarray(present, dtype=bool) & valid
    unknown = present & (tokens == 100)
    lexical = present & ~unknown
    return {'text_present': present, 'text_unknown': unknown,
            'text_lexical_available': lexical,
            'text_observed': present if policy == 'retain' else lexical}
