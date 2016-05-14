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

## What does it checks?

Currently:

- **The spaces before and after operators**

- **In a declaration, if the type is on the same line than the declarator**

```
antoine@hp-blinux:~/cs/1tbs$ cat test.c
/*
** test.c for  in /home/antoine
**
** Made by antoine
** Login   <antoine@epitech.net>
**
** Started on  Fri May 13 12:42:49 2016 antoine
** Last update Fri May 13 12:42:49 2016 antoine
*/

int main()
{
  int
    a;
}
antoine@hp-blinux:~/cs/1tbs$ ./1tbs.py test.c
test.c:13:3: 'int' is not on the same line than 'a'
```

- **The comments**

Optionnally, it can check if the username in a header comment is
valid with the `--header-username` option.

- **The indentation of the preprocessor directives**

- **The function definitions length**

- **The function definitions count in a file**

- **The lines length**

- **The parentheses after `return`**

  TODO: Check for the space after `return`

- **The trailing whitespaces at the end of the lines**

Forbiden statements like `for` and `switch` are not implemented
in the parser. If you use one, it leads to a syntax error.


## Run the tests

Use `python3 -m unittest 1tbs`.

## TODO

A lot of work.

The parser can already process complex declarations like these ones:

```c
int *const *b;
int (*getFunc())(int, int (*)(long));
```

Macros are not implemented, `#define`s are ignored.

Structure and unions are not entirely implemeneted. Bit fields are
not implemented.
