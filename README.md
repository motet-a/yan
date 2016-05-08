# 1TBS — The One True Bocal Style checker

**Work in progress**

**1TBS** stands for "the one true Bocal style". This is a "norminette",
a C brace style checker written in Python designed for the EPITECH style.

But unlike the "official" one, this style checker is not a buch of crappy
regex getting so easily confused.
The 1TBS checker is based on a carefully designed tokenizer and an
authentic hand-written recursive descent parser.

However, the 1TBS checker does not handle macros and preprocessor
directives (except `#include`).

## Run the tests

Use `python3 -m unittest 1tbs`.

## TODO

A lot of work.

The parser can already process complex declarations like these ones:

```
int *const *b;
int (*getFunc())(int, int (*)(long));
```

Macros are not implemented, `#define`s are ignored.

Structure and unions are not entirely implemeneted. Bit fields are
not implemented.
