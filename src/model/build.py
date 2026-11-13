"""Rebuild the model: training and evaluation, then the score-to-counts step.

Run this instead of the two steps separately, so model_meta.json always has the counts block.

Usage: python -u -m src.model.build
"""
from src.model import counts, train


def main():
    train.main()
    print()
    counts.main()


if __name__ == "__main__":
    main()
