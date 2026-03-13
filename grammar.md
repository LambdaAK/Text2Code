# Target Language Grammar

## Syntax

```
e ::= x (variable)
    | n (integer)
    | true | false
    | ( e )                    (parentheses)
    | - e                      (unary negation)
    | e + e
    | e - e
    | e * e
    | e < e
    | e > e
    | e == e
    | e != e
    | e && e
    | e || e
    | if e then e else e
    | let x = e in e
    | lambda x . e
    | e e                     (function application)
```

## Operator Precedence

From **lowest** (binds loosest) to **highest** (binds tightest):

| Precedence | Operator(s) | Associativity |
|------------|-------------|---------------|
| 1 (lowest) | `\|\|`      | left          |
| 2          | `&&`        | left          |
| 3          | `==`, `!=`, `<`, `>` | left  |
| 4          | `+`, `-`    | left          |
| 5          | `*`         | left          |
| 6          | `-` (unary) | —             |
| 7 (highest)| function application `e e` | left |

**Examples:**
- `a || b && c`  →  `a || (b && c)`
- `x + y * z`    →  `x + (y * z)`
- `-a + b`       →  `(-a) + b`
- `f x y`        →  `(f x) y`  (application is left-associative)
