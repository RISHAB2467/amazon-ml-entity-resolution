import argparse
import math

import pandas as pd


BETA = 0.5


def f05(precision, recall):
    if precision == 0 and recall == 0:
        return 0.0

    return (
        (1 + BETA ** 2)
        * precision
        * recall
        / ((BETA ** 2 * precision) + recall)
    )


def score_one(true_ids, predicted_ids):
    true_set = set(true_ids)
    pred_set = set(predicted_ids)

    if not true_set:
        return 1.0 if not pred_set else 0.0

    if not pred_set:
        return 0.0

    tp = len(true_set & pred_set)
    precision = tp / len(pred_set)
    recall = tp / len(true_set)

    return f05(precision, recall)


def macro_f05(ground_truth, predictions):
    """
    ground_truth:
        dict {s1_id: set(true candidate IDs)}

    predictions:
        dict {s1_id: set(predicted IDs)}

    IMPORTANT:
    Every S1 in ground_truth is evaluated, including S1s
    with zero true matches.
    """

    all_s1 = set(ground_truth) | set(predictions)

    if not all_s1:
        return 0.0

    scores = []

    for s1_id in all_s1:
        true_ids = ground_truth.get(s1_id, set())
        pred_ids = predictions.get(s1_id, set())

        scores.append(score_one(true_ids, pred_ids))

    return sum(scores) / len(scores)


def load_ground_truth(path):
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    result = {}

    for _, row in df.iterrows():
        s1_id = row["source1_entity_id"]
        value = row["matched_entity_ids"]

        if value.strip() == "":
            result[s1_id] = set()
        else:
            result[s1_id] = {
                x.strip()
                for x in value.split(",")
                if x.strip()
            }

    return result


def load_predictions(path):
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    result = {}

    for _, row in df.iterrows():
        s1_id = row["source1_entity_id"]
        value = row["matched_entity_ids"]

        if value.strip() == "":
            result[s1_id] = set()
        else:
            result[s1_id] = {
                x.strip()
                for x in value.split(",")
                if x.strip()
            }

    return result


def run_sanity_tests():
    # One true match, perfect prediction.
    assert score_one({"A"}, {"A"}) == 1.0

    # One true match, prediction empty.
    assert score_one({"A"}, set()) == 0.0

    # No true matches, empty prediction.
    assert score_one(set(), set()) == 1.0

    # No true matches, false prediction.
    assert score_one(set(), {"A"}) == 0.0

    # True {A}, prediction {A,B}.
    value = score_one({"A"}, {"A", "B"})
    assert abs(value - 0.5555555555555556) < 1e-9

    # True {A,B}, prediction {A}.
    value = score_one({"A", "B"}, {"A"})
    assert abs(value - 0.8333333333333334) < 1e-9

    print("All scorer sanity tests passed.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ground-truth")
    parser.add_argument("--predictions")

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run scorer sanity tests."
    )

    args = parser.parse_args()

    if args.test:
        run_sanity_tests()

    if args.ground_truth and args.predictions:
        gt = load_ground_truth(args.ground_truth)
        pred = load_predictions(args.predictions)

        score = macro_f05(gt, pred)

        print(f"Macro F0.5: {score:.6f}")

    elif not args.test:
        parser.error(
            "Provide --ground-truth and --predictions, "
            "or use --test."
        )


if __name__ == "__main__":
    main()