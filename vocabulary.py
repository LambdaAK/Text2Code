"""English vocabulary for natural language descriptions of the target language."""

# Digits 0-9 (as strings for tokenization)
DIGITS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# Number words
NUMBER_WORDS = [
    "zero", "one", "two", "three", "four", "five",
    "six", "seven", "eight", "nine", "ten",
]

# Full word vocabulary (alphabetized)
WORDS = [
    "a", "add", "an", "and", "apply", "are", "be", "below", "bind", "both", "by",
    "calculate", "call", "compute", "different", "define", "double", "eight", "else",
    "equal", "equals", "evaluate", "expression", "false", "five", "four", "from",
    "function", "get", "give", "greater", "half", "if", "in", "integer", "is", "it",
    "lambda", "larger", "less", "let", "minus", "more", "multiply", "negative",
    "negate", "negated", "nine", "not", "number", "of", "one", "or", "otherwise",
    "plus", "positive", "return", "returns", "result", "same", "set", "seven", "six",
    "smaller", "subtract", "sum", "take", "takes", "ten", "than", "the", "then",
    "three", "times", "to", "triple", "true", "two", "unequal", "value", "variable",
    "when", "whenever", "which", "with", "zero",
]

# Complete token list: digits + words
VOCABULARY = DIGITS + WORDS
