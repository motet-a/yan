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

**The spaces before and after operators**

```c
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
  return (3 +4);
}
antoine@hp-blinux:~/cs/1tbs$ ./1tbs.py test.c
test.c:13:13: Expected 1 space(s) between '+' and '4'
```

**In a declaration, if the type is on the same line than the declarator**

```sh
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

**The comments**

```sh
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
/*
** This is forbidden
*/
  return (0);
}
antoine@hp-blinux:~/cs/1tbs$ ./1tbs.py test.c
test.c:13:1: Comment inside a function
```

```sh
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

/*
 * This is bad
 */

/* This is forbidden */
int main()
{
  return (0);
}
antoine@hp-blinux:~/cs/1tbs$ ./1tbs.py test.c
test.c:11:1: A comment must end with '*/'
test.c:11:1: The comment lines should start with '**'
test.c:15:1: A comment must be at least 3 lines long
```

Optionnally:

```sh
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
  return (0);
}
antoine@hp-blinux:~/cs/1tbs$ ./1tbs.py --header-username test.c
test.c:1:1: Not a valid EPITECH username (was 'antoine')
test.c:1:1: Not a valid EPITECH username (was 'antoine')
test.c:1:1: Not a valid EPITECH username (was 'antoine')
```

**The indentation of the preprocessor directives**

**The function definitions length**

**The function definitions count in a file**

**The lines length**

**The parentheses after `return`**

```sh
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
  return 0;
}
antoine@hp-blinux:~/cs/1tbs$ ./1tbs.py test.c
test.c:13:8: No paretheses after 'return'
antoine@hp-blinux:~/cs/1tbs$
```

TODO: Check for the space after `return`

**The trailing whitespaces at the end of the lines**



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
