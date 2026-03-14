"""Data generator for (natural language, code) pairs."""

import argparse
import json
import random
from dataclasses import dataclass
from typing import Iterator, List, Optional

from vocabulary import NUMBER_WORDS

# Variable names available for generation
VARIABLES = ["x", "y", "z", "n", "a", "b", "c", "f", "g"]

INT_MIN, INT_MAX = 0, 9

PRECEDENCE = {
    "||": 1,
    "&&": 2,
    "<": 3, ">": 3, "==": 3, "!=": 3,
    "+": 4, "-": 4,
    "*": 5,
    "neg": 6,
    "app": 7,
}

NUMBER_TO_WORD = {i: w for i, w in enumerate(NUMBER_WORDS)}


@dataclass
class Expr:
    tag: str
    args: tuple

    def __repr__(self):
        return f"Expr({self.tag}, {self.args})"


# ─── Code emission ────────────────────────────────────────────────────────────

def _prec(expr: Expr) -> int:
    if expr.tag == "binop":
        return PRECEDENCE.get(expr.args[0], 0)
    if expr.tag == "unop":
        return PRECEDENCE["neg"]
    if expr.tag == "app":
        return PRECEDENCE["app"]
    if expr.tag in ("if", "let", "lambda"):
        return 0  # always parenthesize when embedded inside operators/application
    return 99  # var, int, bool bind tightest


def _maybe_paren(s: str, expr: Expr, parent_prec: int, right_child: bool = False) -> str:
    """Parenthesize if child binds looser than parent, or same-prec right child
    (since all binary ops are left-associative)."""
    p = _prec(expr)
    if p < parent_prec:
        return f"({s})"
    if p == parent_prec and right_child:
        return f"({s})"
    return s


def expr_to_code(expr: Expr, parent_prec: int = 0) -> str:
    """Emit minimal-parentheses code string from AST."""
    if expr.tag == "var":
        return expr.args[0]
    if expr.tag == "int":
        return str(expr.args[0])
    if expr.tag == "bool":
        return expr.args[0]
    if expr.tag == "unop":
        _, e = expr.args
        inner = expr_to_code(e, PRECEDENCE["neg"])
        return f"-{_maybe_paren(inner, e, PRECEDENCE['neg'])}"
    if expr.tag == "binop":
        op, left, right = expr.args
        prec = PRECEDENCE[op]
        ls = _maybe_paren(expr_to_code(left, prec), left, prec)
        rs = _maybe_paren(expr_to_code(right, prec), right, prec, right_child=True)
        return f"{ls} {op} {rs}"
    if expr.tag == "if":
        cond, thn, els = expr.args
        return (
            f"if {expr_to_code(cond)} then {expr_to_code(thn)} else {expr_to_code(els)}"
        )
    if expr.tag == "let":
        var, val, body = expr.args
        return f"let {var} = {expr_to_code(val)} in {expr_to_code(body)}"
    if expr.tag == "lambda":
        var, body = expr.args
        return f"lambda {var} . {expr_to_code(body)}"
    if expr.tag == "app":
        f, arg = expr.args
        prec = PRECEDENCE["app"]
        fs = _maybe_paren(expr_to_code(f, prec), f, prec)
        ags = _maybe_paren(expr_to_code(arg, prec), arg, prec, right_child=True)
        return f"{fs} {ags}"
    raise ValueError(f"Unknown tag: {expr.tag}")


# ─── Type-directed expression generation ──────────────────────────────────────

def _wc(options):
    """Weighted random choice from [(item, weight), ...]. Zero-weight items skipped."""
    filtered = [(item, w) for item, w in options if w > 0]
    items, weights = zip(*filtered)
    return random.choices(items, weights=weights, k=1)[0]


def _add_var(vars: List[str], var: str) -> List[str]:
    """Extend scope with a new variable (no-op if already present)."""
    return vars + [var] if var not in vars else vars


def _gen_func(depth: int, max_depth: int, vars: List[str]) -> Expr:
    """Generate the function position of an application.

    Usually a variable; ~25% of the time a lambda when depth allows, so the
    model sees direct beta-redex patterns like (lambda x . x + 1) 5.
    """
    if depth < max_depth - 1 and random.random() < 0.25:
        var = random.choice(VARIABLES)
        body = _gen_any(depth + 1, max_depth, _add_var(vars, var))
        return Expr("lambda", (var, body))
    return Expr("var", (random.choice(vars or VARIABLES),))


def _int_leaf(vars: List[str]) -> Expr:
    if vars and random.random() < 0.4:
        return Expr("var", (random.choice(vars),))
    return Expr("int", (random.randint(INT_MIN, INT_MAX),))


def _bool_leaf(vars: List[str]) -> Expr:
    # Vars-as-bool is less common — keeps bool positions mostly literal
    if vars and random.random() < 0.15:
        return Expr("var", (random.choice(vars),))
    return Expr("bool", (random.choice(["true", "false"]),))


def _gen_int(depth: int, max_depth: int, vars: List[str]) -> Expr:
    """Generate an int-typed expression."""
    if depth >= max_depth or (depth > 0 and random.random() < 0.35):
        return _int_leaf(vars)

    d = depth + 1
    choice = _wc([
        ("int",    2),
        ("var",    2 if vars else 0),
        ("neg",    1),
        ("+",      3),
        ("-",      2),
        ("*",      2),
        ("if",     1),
        ("let",    1),
        ("app",    1),
    ])

    if choice == "int":
        return Expr("int", (random.randint(INT_MIN, INT_MAX),))
    if choice == "var":
        return Expr("var", (random.choice(vars),))
    if choice == "neg":
        return Expr("unop", ("-", _gen_int(d, max_depth, vars)))
    if choice == "+":
        return Expr("binop", ("+", _gen_int(d, max_depth, vars), _gen_int(d, max_depth, vars)))
    if choice == "-":
        return Expr("binop", ("-", _gen_int(d, max_depth, vars), _gen_int(d, max_depth, vars)))
    if choice == "*":
        return Expr("binop", ("*", _gen_int(d, max_depth, vars), _gen_int(d, max_depth, vars)))
    if choice == "if":
        cond = _gen_bool(d, max_depth, vars)
        return Expr("if", (cond, _gen_int(d, max_depth, vars), _gen_int(d, max_depth, vars)))
    if choice == "let":
        var = random.choice(vars or VARIABLES)
        val = _gen_any(d, max_depth, vars)
        body = _gen_int(d, max_depth, _add_var(vars, var))  # var IS in scope for body
        return Expr("let", (var, val, body))
    if choice == "app":
        return Expr("app", (_gen_func(d, max_depth, vars), _gen_any(d, max_depth, vars)))
    return _int_leaf(vars)


def _gen_bool(depth: int, max_depth: int, vars: List[str]) -> Expr:
    """Generate a bool-typed expression."""
    if depth >= max_depth or (depth > 0 and random.random() < 0.35):
        return _bool_leaf(vars)

    d = depth + 1
    choice = _wc([
        ("bool",   1),
        ("<",      3),
        (">",      2),
        ("==",     3),
        ("!=",     2),
        ("&&",     2),
        ("||",     2),
        ("if",     1),
        ("let",    1),
        ("app",    1),
    ])

    if choice == "bool":
        return _bool_leaf(vars)
    if choice in ("<", ">", "==", "!="):
        return Expr("binop", (choice, _gen_int(d, max_depth, vars), _gen_int(d, max_depth, vars)))
    if choice == "&&":
        return Expr("binop", ("&&", _gen_bool(d, max_depth, vars), _gen_bool(d, max_depth, vars)))
    if choice == "||":
        return Expr("binop", ("||", _gen_bool(d, max_depth, vars), _gen_bool(d, max_depth, vars)))
    if choice == "if":
        cond = _gen_bool(d, max_depth, vars)
        return Expr("if", (cond, _gen_bool(d, max_depth, vars), _gen_bool(d, max_depth, vars)))
    if choice == "let":
        var = random.choice(vars or VARIABLES)
        val = _gen_any(d, max_depth, vars)
        body = _gen_bool(d, max_depth, _add_var(vars, var))
        return Expr("let", (var, val, body))
    if choice == "app":
        return Expr("app", (_gen_func(d, max_depth, vars), _gen_any(d, max_depth, vars)))
    return _bool_leaf(vars)


def _gen_any(depth: int, max_depth: int, vars: List[str]) -> Expr:
    """Generate an expression of any type (top-level or unconstrained positions)."""
    if depth >= max_depth or (depth > 0 and random.random() < 0.35):
        return random.choice([_int_leaf(vars), _bool_leaf(vars)])

    d = depth + 1
    choice = _wc([
        ("int_expr",  4),
        ("bool_expr", 3),
        ("lambda",    1),
        ("app",       1),
        ("if",        1),
        ("let",       1),
    ])

    if choice == "int_expr":
        return _gen_int(depth, max_depth, vars)
    if choice == "bool_expr":
        return _gen_bool(depth, max_depth, vars)
    if choice == "lambda":
        var = random.choice(VARIABLES)
        body = _gen_any(d, max_depth, _add_var(vars, var))  # var IS in scope for body
        return Expr("lambda", (var, body))
    if choice == "app":
        return Expr("app", (_gen_func(d, max_depth, vars), _gen_any(d, max_depth, vars)))
    if choice == "if":
        cond = _gen_bool(d, max_depth, vars)
        return Expr("if", (cond, _gen_any(d, max_depth, vars), _gen_any(d, max_depth, vars)))
    if choice == "let":
        var = random.choice(vars or VARIABLES)
        val = _gen_any(d, max_depth, vars)
        body = _gen_any(d, max_depth, _add_var(vars, var))
        return Expr("let", (var, val, body))
    return _int_leaf(vars)


def generate_expression(
    depth: int = 0,
    max_depth: int = 3,
    available_vars: Optional[List[str]] = None,
) -> Expr:
    if available_vars is None:
        available_vars = list(VARIABLES)
    return _gen_any(depth, max_depth, available_vars)


# ─── Natural language generation ──────────────────────────────────────────────

def _is_atom(expr: Expr) -> bool:
    return expr.tag in ("var", "int", "bool")


def _int_word(n: int) -> str:
    if random.random() < 0.5:
        return NUMBER_TO_WORD.get(n, str(n))
    return str(n)


def _describe_var(name: str) -> str:
    """Occasionally add filler around a variable name."""
    if random.random() < 0.12:
        return random.choice([
            f"the value of {name}",
            f"the variable {name}",
        ])
    return name


def _describe(expr: Expr) -> str:
    """Recursively produce a natural language description of the expression.

    Disambiguation strategy: when one or both children of a binary op are
    non-atomic (compound sub-expressions), we use structured "the X of A and B"
    phrasing, which clearly groups A and B as the two operands. Infix forms are
    used freely when both children are atomic (variables or literals), where
    there is no ambiguity.
    """
    # ── Atoms ────────────────────────────────────────────────────────────────
    if expr.tag == "var":
        return _describe_var(expr.args[0])
    if expr.tag == "int":
        return _int_word(expr.args[0])
    if expr.tag == "bool":
        return expr.args[0]

    # ── Unary negation ────────────────────────────────────────────────────────
    if expr.tag == "unop":
        inner = _describe(expr.args[1])
        return random.choice([
            f"negate {inner}",
            f"the negation of {inner}",
            f"negative {inner}",
            f"negated {inner}",
            f"minus {inner}",
            f"the negative of {inner}",
        ])

    # ── Binary operators ──────────────────────────────────────────────────────
    if expr.tag == "binop":
        op, left, right = expr.args
        l = _describe(left)
        r = _describe(right)
        both_atom = _is_atom(left) and _is_atom(right)

        if op == "+":
            # Commutative: occasionally swap operand order in NL only
            if random.random() < 0.2:
                l, r = r, l
            if both_atom:
                return random.choice([
                    f"{l} plus {r}",
                    f"add {l} and {r}",
                    f"the sum of {l} and {r}",
                    f"{l} added to {r}",
                    f"sum of {l} and {r}",
                    f"{l} and {r} added together",
                ])
            return random.choice([
                f"the sum of {l} and {r}",
                f"add {l} and {r}",
                f"sum of {l} and {r}",
            ])

        if op == "-":
            if both_atom:
                return random.choice([
                    f"{l} minus {r}",
                    f"subtract {r} from {l}",
                    f"the difference of {l} and {r}",
                    f"{l} decreased by {r}",
                    f"take {r} from {l}",
                    f"{l} take away {r}",
                ])
            return random.choice([
                f"subtract {r} from {l}",
                f"the difference of {l} and {r}",
                f"{l} decreased by {r}",
            ])

        if op == "*":
            # Commutative: occasionally swap operand order in NL only
            if random.random() < 0.2:
                l, r = r, l
            if both_atom:
                return random.choice([
                    f"{l} times {r}",
                    f"multiply {l} and {r}",
                    f"the product of {l} and {r}",
                    f"{l} multiplied by {r}",
                    f"multiply {l} by {r}",
                    f"the product of {l} times {r}",
                ])
            return random.choice([
                f"the product of {l} and {r}",
                f"multiply {l} by {r}",
                f"multiply {l} and {r}",
            ])

        if op == "<":
            if both_atom:
                return random.choice([
                    f"{l} less than {r}",
                    f"{l} is less than {r}",
                    f"{l} is smaller than {r}",
                    f"{l} below {r}",
                    f"{l} is below {r}",
                ])
            return random.choice([
                f"{l} is less than {r}",
                f"{l} less than {r}",
                f"{l} is smaller than {r}",
            ])

        if op == ">":
            if both_atom:
                return random.choice([
                    f"{l} greater than {r}",
                    f"{l} is greater than {r}",
                    f"{l} is larger than {r}",
                    f"{l} above {r}",
                    f"{l} is above {r}",
                    f"{l} is more than {r}",
                ])
            return random.choice([
                f"{l} is greater than {r}",
                f"{l} greater than {r}",
                f"{l} is larger than {r}",
            ])

        if op == "==":
            if both_atom:
                return random.choice([
                    f"{l} equals {r}",
                    f"{l} is equal to {r}",
                    f"{l} is the same as {r}",
                    f"{l} equal to {r}",
                    f"{l} and {r} are equal",
                    f"{r} equals {l}",
                ])
            return random.choice([
                f"{l} equals {r}",
                f"{l} is equal to {r}",
                f"{l} is the same as {r}",
            ])

        if op == "!=":
            if both_atom:
                return random.choice([
                    f"{l} not equal to {r}",
                    f"{l} is not equal to {r}",
                    f"{l} is different from {r}",
                    f"{l} unequal to {r}",
                    f"{l} and {r} are different",
                    f"{l} differs from {r}",
                ])
            return random.choice([
                f"{l} is not equal to {r}",
                f"{l} not equal to {r}",
                f"{l} is different from {r}",
            ])

        if op == "&&":
            if both_atom:
                return random.choice([
                    f"{l} and {r}",
                    f"both {l} and {r}",
                    f"{l} and also {r}",
                    f"{l} as well as {r}",
                ])
            return random.choice([
                f"{l} and {r}",
                f"both {l} and {r}",
                f"{l} and also {r}",
            ])

        if op == "||":
            if both_atom:
                return random.choice([
                    f"{l} or {r}",
                    f"either {l} or {r}",
                    f"{l} or else {r}",
                    f"{l} or alternatively {r}",
                ])
            return random.choice([
                f"{l} or {r}",
                f"either {l} or {r}",
                f"{l} or else {r}",
            ])

    # ── Conditional ───────────────────────────────────────────────────────────
    if expr.tag == "if":
        cond, thn, els = expr.args
        c = _describe(cond)
        t = _describe(thn)
        e = _describe(els)
        return random.choice([
            f"if {c} then {t} else {e}",
            f"when {c}, {t}, otherwise {e}",
            f"if {c} then {t}, otherwise {e}",
            f"if {c}, {t} else {e}",
            f"in case {c}, {t}, otherwise {e}",
            f"when {c} then {t} else {e}",
        ])

    # ── Let binding ───────────────────────────────────────────────────────────
    if expr.tag == "let":
        var, val, body = expr.args
        v = _describe(val)
        b = _describe(body)
        return random.choice([
            f"let {var} be {v} in {b}",
            f"let {var} equal {v} in {b}",
            f"define {var} as {v} in {b}",
            f"set {var} to {v} in {b}",
            f"where {var} is {v}, {b}",
            f"with {var} equal to {v}, {b}",
            f"given {var} equals {v}, {b}",
        ])

    # ── Lambda ────────────────────────────────────────────────────────────────
    if expr.tag == "lambda":
        var, body = expr.args
        b = _describe(body)
        return random.choice([
            f"a function that takes {var} and returns {b}",
            f"lambda {var} . {b}",
            f"the function taking {var} and returning {b}",
            f"a function of {var} that returns {b}",
            f"function that takes {var} and gives {b}",
            f"takes {var} and returns {b}",
        ])

    # ── Application ───────────────────────────────────────────────────────────
    if expr.tag == "app":
        f, arg = expr.args
        fn = _describe(f)
        a = _describe(arg)
        return random.choice([
            f"apply {fn} to {a}",
            f"call {fn} with {a}",
            f"{fn} applied to {a}",
            f"{fn} of {a}",
            f"call {fn} on {a}",
        ])

    raise ValueError(f"Unknown expr tag: {expr.tag}")


# ─── Dataset generation ───────────────────────────────────────────────────────

def _sample_depth(max_depth: int) -> int:
    """Curriculum sampling: shallower depths are more likely.

    For max_depth=3: P(1)≈50%, P(2)≈33%, P(3)≈17%
    This ensures the model sees many simple examples before complex ones.
    """
    depths = list(range(1, max_depth + 1))
    weights = [max_depth - d + 1 for d in depths]
    return random.choices(depths, weights=weights, k=1)[0]


def _maybe_wrap(nl: str) -> str:
    """Occasionally prefix the description with a meta-verb (~15% of samples)."""
    if random.random() < 0.15:
        verb = random.choice(["compute", "evaluate", "calculate", "return"])
        return f"{verb} {nl}"
    return nl


def generate_pair(max_depth: int = 3, seed: Optional[int] = None) -> tuple:
    """Generate a single (natural_language, code) pair."""
    if seed is not None:
        random.seed(seed)
    depth = _sample_depth(max_depth)
    expr = generate_expression(0, depth)
    return (_maybe_wrap(_describe(expr)), expr_to_code(expr))


def generate_dataset(
    n: int,
    max_depth: int = 3,
    seed: Optional[int] = None,
) -> List[tuple]:
    """Generate n unique-NL (natural_language, code) pairs.

    Deduplicates by NL string. Retries up to 10x the requested count before
    giving up, printing a warning if fewer unique pairs were found.
    """
    if seed is not None:
        random.seed(seed)

    seen: set = set()
    pairs: List[tuple] = []
    max_attempts = n * 10

    for _ in range(max_attempts):
        if len(pairs) == n:
            break
        nl, code = generate_pair(max_depth=max_depth)
        if nl not in seen:
            seen.add(nl)
            pairs.append((nl, code))

    if len(pairs) < n:
        print(f"Warning: only {len(pairs)} unique pairs found after {max_attempts} attempts "
              f"(try increasing -d or reducing -n)")
    return pairs


def split_dataset(
    pairs: List[tuple],
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: Optional[int] = None,
) -> tuple:
    """Randomly split pairs into (train, val, test) lists."""
    if seed is not None:
        random.seed(seed)
    data = list(pairs)
    random.shuffle(data)
    n = len(data)
    n_test = max(1, int(n * test_ratio))
    n_val  = max(1, int(n * val_ratio))
    test  = data[:n_test]
    val   = data[n_test:n_test + n_val]
    train = data[n_test + n_val:]
    return train, val, test


def _write_jsonl(path: str, pairs: List[tuple]) -> None:
    with open(path, "w") as f:
        for nl, code in pairs:
            f.write(json.dumps({"nl": nl, "code": code}) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate (NL, code) pairs")
    parser.add_argument("-n", "--num", type=int, default=10,
                        help="Number of unique pairs to generate")
    parser.add_argument("-d", "--max-depth", type=int, default=3,
                        help="Max expression depth")
    parser.add_argument("-o", "--output", type=str,
                        help="Output file (JSONL). With --split, used as path prefix.")
    parser.add_argument("-s", "--seed", type=int, help="Random seed")
    parser.add_argument("--split", action="store_true",
                        help="Split output into train/val/test files")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Fraction of data for validation (default: 0.1)")
    parser.add_argument("--test-ratio", type=float, default=0.1,
                        help="Fraction of data for test (default: 0.1)")
    args = parser.parse_args()

    pairs = generate_dataset(args.num, max_depth=args.max_depth, seed=args.seed)

    if args.split:
        train, val, test = split_dataset(
            pairs,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        if args.output:
            # Strip .jsonl suffix if present to use as prefix
            prefix = args.output[:-6] if args.output.endswith(".jsonl") else args.output
            paths = {
                "train": f"{prefix}_train.jsonl",
                "val":   f"{prefix}_val.jsonl",
                "test":  f"{prefix}_test.jsonl",
            }
        else:
            paths = {"train": "train.jsonl", "val": "val.jsonl", "test": "test.jsonl"}
        for split_name, split_pairs in [("train", train), ("val", val), ("test", test)]:
            _write_jsonl(paths[split_name], split_pairs)
            print(f"Wrote {len(split_pairs):>6} pairs → {paths[split_name]}")
    elif args.output:
        _write_jsonl(args.output, pairs)
        print(f"Wrote {len(pairs)} pairs to {args.output}")
    else:
        print("Sample (natural language, code) pairs:\n")
        for i, (nl, code) in enumerate(pairs, 1):
            print(f"{i}. NL:   {nl}")
            print(f"   Code: {code}\n")


if __name__ == "__main__":
    main()
