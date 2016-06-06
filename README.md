# Yan — Yet another norminette

**Work in progress**

**Yan** stands for "Yet another *norminette*". This is a C brace style
checker written in Python designed for the EPITECH style.

But unlike the "official" one, this style checker is not a buch of crappy
regex getting so easily confused.
Yan is based on a carefully designed tokenizer and an authentic
hand-written recursive descent parser.

However, Yan does not handle macros and preprocessor directives (except
`#include`).



## What does it checks?

A lot of rules of the *EPITECH style* are checked. I don't want to list
all the checks here, it's a pain to describe and to read. Here is a short
summary.

A few rules are not implemented yet, and a few rules are not implementable
(they must be checked by a human).

A notable feature of this program is its ability to check the most of
the rules relative to the indentation.

Forbiden statements like `for` and `switch` are not implemented
in the parser. If you use one, it leads to a syntax error.



## Other features

`<stdarg.h>` and `va_arg()` are supported. « Calls » to function-like
macros with a type as argument are supported, ellipsis (`...`) is
supported.

`typedef` is (partially) supported. If Yan fails to analyze a
header containing a type important in your program (say `t_my_type`), you
can add this comment before your first use of `t_my_type`:

```c
/*
** yan typedef t_my_type
*/
```

If Yan crashes on a part of your code, you can indicate it to ignore the
problematic part:

```c
/*
** yan parser off
*/

...some C++ stuff to skip...

/*
** yan parser on
*/
```

Some features of C99 are implemented (e.g. designated initializers).



## Run the tests

Use `python3 -m unittest yan`.



## TODO

Not so much work.

Macros are not implemented, `#define`s are ignored. If you use a macro,
it should look like a function call or an identifier, otherwise it leads
to a syntax error.

Structure and unions are not entirely implemented. Bit fields are
not implemented.
