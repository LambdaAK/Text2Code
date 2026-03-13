# English Vocabulary for Natural Language Descriptions

A constrained vocabulary for describing expressions in the target language. Organized by construct.

---

## Arithmetic

| Word(s) | Use case |
|---------|----------|
| add, plus, sum | `e + e` |
| subtract, minus, take away | `e - e` |
| multiply, times | `e * e` |
| negate, negative, negated | `- e` |
| double | idiom for `e * 2` |
| triple | idiom for `e * 3` |

---

## Comparisons

| Word(s) | Use case |
|---------|----------|
| less than, smaller than, below | `e < e` |
| greater than, bigger than, more than, larger than, above | `e > e` |
| equal, equals, is equal to, same as | `e == e` |
| not equal, not equals, different from, unequal | `e != e` |
| positive | `x > 0` |
| negative | `x < 0` (when describing sign, distinct from unary negate) |
| zero | `== 0` |

---

## Logical

| Word(s) | Use case |
|---------|----------|
| and, both | `e && e` |
| or, either | `e \|\| e` |
| not | for `!e` — *not in grammar; omit or add later* |

---

## Booleans

| Word(s) | Use case |
|---------|----------|
| true | literal |
| false | literal |

---

## Conditionals

| Word(s) | Use case |
|---------|----------|
| if, when, whenever | condition |
| then | consequence |
| else, otherwise | alternative |

---

## Let / Binding

| Word(s) | Use case |
|---------|----------|
| let, define, set, bind | `let x = e` |
| be, equal, to | `x = e` |
| in | `in e` |

---

## Functions

| Word(s) | Use case |
|---------|----------|
| function, lambda | `lambda x . e` |
| take, takes | parameter introduction |
| return, returns | body description |
| apply, call | `e e` |
| to, and | "a function that takes x and returns..." |

---

## Variables & Values

| Word(s) | Use case |
|---------|----------|
| variable, var | generic reference |
| value, result | "the value of x" |
| expression | generic |
| it | pronoun for previous expression |

---

## Numbers (for integer literals)

| Form | Token(s) | Use case |
|------|----------|----------|
| Digits | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 | numeric literals in NL |
| Words | zero, one, two, three, four, five, six, seven, eight, nine | word form of literals |
| Generic | number, integer | generic reference |

---

## Structure / Function Words

| Word(s) | Use case |
|---------|----------|
| the, a, an | articles |
| of | "the sum of x and y" |
| to, from, by, with | prepositions |
| is, are | copula |
| that, which | relative clause |
| compute, calculate, evaluate | meta-verbs |

---

## Complete Token List (Alphabetized)

**Digits:** 0, 1, 2, 3, 4, 5, 6, 7, 8, 9

**Words:**
```
a, add, an, and, apply, are, be, below, bind, both, by, calculate, call, compute,
different, define, double, eight, else, equal, equals, evaluate, expression,
false, five, four, from, function, get, give, greater, half, if, in,
integer, is, it, lambda, larger, less, let, minus, more, multiply,
negative, negate, negated, nine, not, number, of, one, or, otherwise,
plus, positive, return, returns, result, same, set, seven, six, smaller,
subtract, sum, take, takes, ten, than, the, then, three, times, to,
triple, true, two, unequal, value, variable, when, whenever, which,
with, zero
```

---

## Notes

- **Variable names** (x, y, z, n, f, etc.) are used as-is in descriptions.
- **"not"** is listed but `!e` is not in the grammar; omit from generation or add `!e` later.
- **"half"** could support a future division operator; omit for now.
- Consider a **minimum viable subset** for initial data generation (e.g., drop rare words like "triple", "whenever").
- **Plural forms** (adds, returns, etc.) may be needed for third-person; include if generating varied sentence structures.
