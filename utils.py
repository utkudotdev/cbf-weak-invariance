"""Shared argparse types for numeric arguments."""

import argparse


def int_at_least(value: str, *, minimum: int) -> int:
    number = int(value)
    if number < minimum:
        raise argparse.ArgumentTypeError(f"must be an integer >= {minimum}")
    return number


def positive_int(value: str) -> int:
    return int_at_least(value, minimum=1)


def positive_float(value: str) -> float:
    number = float(value)
    if not number > 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def nonnegative_float(value: str) -> float:
    number = float(value)
    if not number >= 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def fraction(value: str) -> float:
    number = float(value)
    if not 0 <= number < 1:
        raise argparse.ArgumentTypeError("must be in [0, 1)")
    return number
