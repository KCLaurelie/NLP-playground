"""
This tool is described in "High Throughput Phenotyping for Dimensional 
Psychopathology in Electronic Health Records" 
  See: doi:10.1016/j.biopsych.2018.01.011.

It was developed and tested against Python 2.7. It is packaged as a single 
script which depends only on the standard library to simplify distribution.

By line, the majority of the file is token lists (per domain, lexical variants,
and stop words). They are used at the end of the file to get phenotype scores.
"""

import re
import sys

import CQH-Dimensional-Phenotyper-master.CQH_input_data


WHITE_SPACE = re.compile(r'^\s+', re.MULTILINE)
SPECIAL = re.compile(r'[^A-Za-z0-9]+', re.MULTILINE)


def clean_text(raw_text):
    text = raw_text.encode("ascii", "ignore")
    cleaner = SPECIAL.sub(" ", text.lower())
    return WHITE_SPACE.sub(" ", cleaner)


def tokenize(text, stopwords, lookup):
    ok_tokens = (tk for tk in text.split() if not tk.isdigit() and len(tk) > 2)
    clean = (tk.strip() for tk in ok_tokens if tk not in stopwords)
    return [lookup[tk] if tk in lookup else tk for tk in clean]


def as_bigrams(tokens):
    return zip(tokens, tokens[1:] + [''])


def count_terms(bigrams, to_count):
    counts = {term: 0 for term in to_count}
    for a, b in bigrams:
        if a in counts:
            counts[a] += 1
        bi_key = "_".join((a, b))
        if bi_key in counts:
            counts[bi_key] += 1
    counts['len'] = len(bigrams)
    return counts


def domain_rollup(scores, domain_map=CQH_input_data.DOMAIN_TOKEN_MAP):
    reduced_scores = {}
    for domain in domain_map:
        included_tks = domain_map[domain]
        # A key error here would mean something went wrong w/ the token maps
        reduced_scores[domain] = [scores[tks] for tks in included_tks]
    return reduced_scores


def domain_density(by_domain):
    density = {}
    for domain in by_domain:
        tks_counts = [1 if tk_ct > 0 else 0 for tk_ct in by_domain[domain]]
        density[domain] = round(sum(tks_counts) / float(len(tks_counts)), 5)
    return density


def count_document(raw_text, stopwords=set(STOP_WORDS),
                   to_count=TOKENS,
                   normalize=TOKEN_LVG_MAP):
    tokens = tokenize(clean_text(raw_text), stopwords, normalize)
    return count_terms(as_bigrams(tokens), to_count)


def reduce_counts(counts):
    counts_by_domain = domain_rollup(counts)
    domain_scores = domain_density(counts_by_domain)

    # Attach some match rate meta data to the density scores
    domain_scores['match_count'] = sum(counts.values()) - counts['len']
    domain_scores['length'] = counts['len']
    return domain_scores


if __name__ == "__main__":
    if sys.stdin.isatty():
        print("CQH Dimensional Phenotyper")
        print("Pipe text to be scored into this script.")
    else:
        raw_text = sys.stdin.read()
        results = count_document(raw_text)
        results = reduce_counts(results)
        tpl = "length,match_count,negative,positive,cognitive,social,arousal_regulatory"
        print(tpl)
        print(','.join([str(results[item]) for item in tpl.split(',')]))
