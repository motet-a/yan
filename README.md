# Yan — Yet another norminette

[![Build Status](https://travis-ci.org/motet-a/yan.svg?branch=master)](https://travis-ci.org/motet-a/yan)

**Work in progress**

**Yan** stands for "Yet another *norminette*". This is a C brace style
checker written in Python designed for the EPITECH style.

But unlike the "official" one, this style checker is not a buch of crappy
regex getting so easily confused.
Yan is based on a carefully designed tokenizer and an authentic
hand-written recursive descent parser.

However, Yan does not handle macros and preprocessor directives (except
`#include`).



## What does it check?

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



## Download

Clone the whole repository or fetch `yan.py` only:

```sh
wget https://raw.githubusercontent.com/motet-a/yan/master/yan.py
chmod +x yan.py
```



## Check

If `PSU_2015_my_printf/` is the directory of your repository, simply
run `yan PSU_2015_my_printf/`.

If you have header files in a separate directory and the paths in your
`#include` statements are not relative, Yan can't find the
headers. You have to specify manually which directories contains
headers with the `-I` option.

If you have a Makefile, Yan can automatically find which directory
contains headers in some cases.



## Run the tests

To run all the tests, use `./test.sh`.

You can run the unit tests with `./test.py` or `python3 -m unittest
test.py`.

The script `test_on_real_projects.sh` fetches a few Git repositories
of EPITECH projects and checks their sources with Yan.



## TODO

Not so much work.

Macros are not implemented, `#define`s are ignored. If you use a macro,
it should look like a function call or an identifier, otherwise it leads
to a syntax error.

Structure and unions are not entirely implemented. Bit fields are
not implemented.
