/*
** __attribute___valid.h for  in /home/antoine
**
** Made by antoine
** Login   <antoine@epitech.net>
**
** Started on  Sat Jun 11 09:55:34 2016 antoine
** Last update Sat Jun 11 09:55:34 2016 antoine
*/

#ifndef __ATTRIBUTE___VALID_H_
# define __ATTRIBUTE___VALID_H_

int     main(__attribute__((unused))int argc);

struct                          s_a
{
  __attribute__((unused))int    a;
  __attribute__((unused)) int   b;
  int __attribute__((unused))   c;
} __attribute__((packed));

#endif /* !__ATTRIBUTE___VALID_H_ */
