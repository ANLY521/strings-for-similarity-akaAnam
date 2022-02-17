from difflib import SequenceMatcher

from nltk import word_tokenize, edit_distance
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.nist_score import sentence_nist
from scipy.stats import pearsonr
import argparse
from util import parse_sts
from sts_nist import symmetrical_nist


def main(sts_data):
    """Calculate pearson correlation between semantic similarity scores and string similarity metrics.
    Data is formatted as in the STS benchmark"""

    # TODO 1: read the dataset; implement in util.py
    texts, labels = parse_sts(sts_data)

    print(f"Found {len(texts)} STS pairs")

    # TODO 2: Calculate each of the the metrics here for each text pair in the dataset
    # HINT: Longest common substring can be complicated. Investigate difflib.SequenceMatcher for a good option.
    score_types = ["NIST", "BLEU", "Word Error Rate", "Longest common substring", "Edit Distance"]

    nist = []
    bleu = []
    wer = []
    lcs = []
    ed = []

    for word in texts:
        t1, t2 = word

        # input tokenized text
        t1_toks = word_tokenize(t1.lower())
        t2_toks = word_tokenize(t2.lower())

        # try / except for each side because of ZeroDivision Error
        # 0.0 is lowest score - give that if ZeroDivision Error

        # nist metric
        try:
            nist_1 = sentence_nist([t1_toks, ], t2_toks)
        except ZeroDivisionError:
            # print(f"\n\n\nno NIST, {i}")
            nist_1 = 0.0

        nist.append(nist_1)

        # bleu metric
        # https://machinelearningmastery.com/calculate-bleu-score-for-text-python/
        # smoothing error: https://www.nltk.org/_modules/nltk/translate/bleu_score.html

        try:
            bleu_m = sentence_bleu([t1_toks,],t2_toks, smoothing_function = SmoothingFunction().method0)
        except ZeroDivisionError:
            bleu_m = 0.0
        bleu.append(bleu_m)

        # wer metric
        # formula using the edit dist function

        wer_m = edit_distance(t1_toks,t2_toks)/(len(t1_toks) + len(t2_toks))
        wer.append(wer_m)

        # LCR metric
        lcs_m = SequenceMatcher(None,t1,t2).ratio()
        lcs.append(lcs_m)

        # edit distance metric

        ed_m = edit_distance(t1,t2)
        ed.append(ed_m)


    all_scores = []
    all_scores.append(nist)
    all_scores.append(bleu)
    all_scores.append(wer)
    all_scores.append(lcs)
    all_scores.append(ed)

    #TODO 3: Calculate pearson r between each metric and the STS labels and report in the README.
    # Sample code to print results. You can alter the printing as you see fit. It is most important to put the results
    # in a table in the README
    print(f"Semantic textual similarity for {sts_data}\n")
    for metric_name in range(0,len(score_types)):
        score = pearsonr(all_scores[metric_name], labels)[0]
        print(f"{score_types[metric_name]} correlation: {score:.03f}")

    # TODO 4: Complete writeup as specified by TODOs in README (describe metrics; show usage)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sts_data", type=str, default="stsbenchmark/sts-dev.csv",
                        help="tab separated sts data in benchmark format")
    args = parser.parse_args()

    main(args.sts_data)

