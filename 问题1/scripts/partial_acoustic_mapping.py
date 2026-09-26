"""Preserve valid local paths while isolating structurally invalid word intervals.

No endpoint interpolation or label substitution is allowed. Structural validity
does not certify that the target word was actually spoken.
"""
import math
from acoustic_mapping import map_word_tier


def map_partial_word_tier(source_words, entries, offset_s, duration_s):
    entries = [(float(a), float(b), str(t)) for a, b, t in entries if str(t).strip()]
    # Use a known monotone synthetic clock only to obtain lexical row ownership.
    # Synthetic times are discarded and cannot reach output intervals.
    surrogate = [[i, i + 1, label] for i, (_, _, label) in enumerate(entries)]
    words, status = map_word_tier(source_words, surrogate, 0, max(len(entries), 1))
    if status != 'mapped':
        return words, status
    bad = set()
    for i, (a, b, _) in enumerate(entries):
        if not (math.isfinite(a) and math.isfinite(b) and 0 <= a < b <= duration_s + .001 and a < duration_s):
            bad.add(i)
    valid_rows = [i for i in range(len(entries)) if i not in bad]
    for left, right in zip(valid_rows, valid_rows[1:]):
        if entries[right][0] < entries[left][1] - 1e-7:
            bad.update([left, right])
    for word in words:
        rows = word.get('tool_word_rows', [])
        word['interval_s'] = None
        if not rows:
            continue
        if any(r in bad for r in rows):
            word['status'] = 'invalid_local_interval'
            continue
        a, b = entries[rows[0]][0], min(entries[rows[-1]][1], duration_s)
        word['interval_s'] = [offset_s + a, offset_s + b]
        word['status'] = 'structurally_valid_candidate'
    count = sum(w['interval_s'] is not None for w in words)
    return words, 'mapped' if count == len(words) else 'partially_mapped' if count else 'no_valid_intervals'
