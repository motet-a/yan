"""
Unit tests for Yan.
"""
# pylint: disable=missing-docstring

import unittest
import yan
from yan import (lex,
                 NSyntaxError,
                 Position,
                 parse,
                 parse_expr,
                 parse_statement)


class TestPosition(unittest.TestCase):
    def test_begin_position(self):
        pos = Position('abcd')
        self.assertEqual(pos.file_name, 'abcd')
        self.assertEqual(pos.index, 0)
        self.assertEqual(pos.line, 1)
        self.assertEqual(pos.column, 1)
        self.assertEqual(str(pos), 'abcd:1:1')

    def test_position(self):
        pos = Position('abcd', 2, 3, 4)
        self.assertEqual(pos.index, 2)
        self.assertEqual(pos.line, 3)
        self.assertEqual(pos.column, 4)
        self.assertEqual(str(pos), 'abcd:3:4')


class TestLexer(unittest.TestCase):
    def assertLexEqual(self, source, expected):
        tokens = lex(source)
        self.assertEqual(''.join(repr(t) for t in tokens), expected)

    def assertTokenEqual(self, source, kind, string):
        assert kind in yan.TOKEN_KINDS
        tokens = lex(source)
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0].kind, kind)
        self.assertEqual(tokens[0].string, string)

    def test_skip_spaces(self):
        self.assertLexEqual('', '')
        self.assertLexEqual('  ', '')

    def test_new_line(self):
        self.assertLexEqual('   \n"abc"',
                            '<Token kind=string, string=\'"abc"\', '
                            'begin=<unknown file>:2:1, '
                            'end=<unknown file>:2:5>')

    def test_string(self):
        self.assertLexEqual('"abc"',
                            '<Token kind=string, string=\'"abc"\', '
                            'begin=<unknown file>:1:1, '
                            'end=<unknown file>:1:5>')
        self.assertLexEqual('  "abc"',
                            '<Token kind=string, string=\'"abc"\', '
                            'begin=<unknown file>:1:3, '
                            'end=<unknown file>:1:7>')
        self.assertLexEqual('""',
                            '<Token kind=string, string=\'""\', '
                            'begin=<unknown file>:1:1, '
                            'end=<unknown file>:1:2>')

    def test_multiple_strings(self):
        self.assertLexEqual('"" ""',
                            '<Token kind=string, string=\'""\', '
                            'begin=<unknown file>:1:1, '
                            'end=<unknown file>:1:2>'
                            '<Token kind=string, string=\'""\', '
                            'begin=<unknown file>:1:4, '
                            'end=<unknown file>:1:5>')

    def test_string_error(self):
        with self.assertRaises(NSyntaxError):
            lex('"abc')
        with self.assertRaises(NSyntaxError):
            lex('"\n"')
        with self.assertRaises(NSyntaxError):
            lex('"')

    def test_string_escape(self):
        self.assertTokenEqual(r'"\0"', 'string', r'"\0"')
        self.assertTokenEqual(r'"\n"', 'string', r'"\n"')
        with self.assertRaises(NSyntaxError):
            lex(r'"\"')

    def test_character(self):
        self.assertTokenEqual("'a'", 'character', "'a'")
        self.assertTokenEqual(r"'\0'", 'character', r"'\0'")
        self.assertTokenEqual(r"'\n'", 'character', r"'\n'")

    def test_keyword(self):
        self.assertLexEqual('int',
                            '<Token kind=keyword, string=\'int\', '
                            'begin=<unknown file>:1:1, '
                            'end=<unknown file>:1:3>')

    def test_identifier(self):
        self.assertLexEqual('_Abcde_F',
                            '<Token kind=identifier, string=\'_Abcde_F\', '
                            'begin=<unknown file>:1:1, '
                            'end=<unknown file>:1:8>')

    def test_integer_dec(self):
        self.assertTokenEqual('0', 'integer', '0')
        self.assertTokenEqual('0777', 'integer', '0777')

    def test_integer_hex(self):
        self.assertTokenEqual('0x10', 'integer', '0x10')
        self.assertTokenEqual('0xcafebabe', 'integer', '0xcafebabe')

    def test_integer_ul(self):
        self.assertTokenEqual('0u', 'integer', '0u')
        self.assertTokenEqual('0U', 'integer', '0U')
        self.assertTokenEqual('0L', 'integer', '0L')
        self.assertTokenEqual('0Ll', 'integer', '0Ll')
        self.assertTokenEqual('0uL', 'integer', '0uL')

    def test_float(self):
        self.assertTokenEqual('.0', 'float', '.0')
        self.assertTokenEqual('0.', 'float', '0.')
        self.assertTokenEqual('.0l', 'float', '.0l')
        self.assertTokenEqual('0.l', 'float', '0.l')
        self.assertTokenEqual('0.0', 'float', '0.0')

    def test_comment(self):
        self.assertTokenEqual('/**/', 'comment', '/**/')
        self.assertTokenEqual('/*\n*/', 'comment', '/*\n*/')
        self.assertTokenEqual('/*abc*/', 'comment', '/*abc*/')

    def test_comment_and_string(self):
        self.assertTokenEqual('/*"abc"*/', 'comment', '/*"abc"*/')
        self.assertTokenEqual('"/**/"', 'string', '"/**/"')

    def test_sign(self):
        self.assertTokenEqual('++', 'sign', '++')
        self.assertTokenEqual('.', 'sign', '.')
        self.assertTokenEqual('...', 'sign', '...')
        self.assertTokenEqual('>>', 'sign', '>>')
        self.assertTokenEqual('->', 'sign', '->')
        self.assertTokenEqual('-', 'sign', '-')

    def test_directive(self):
        self.assertTokenEqual('#ifdef a', 'directive', '#ifdef a')
        self.assertTokenEqual('\n#ifdef a', 'directive', '#ifdef a')
        self.assertTokenEqual('\n  #  include <a> \n ',
                              'directive', '  #  include <a> ')
        self.assertTokenEqual('\n  #  include "a" \n ',
                              'directive', '  #  include "a" ')
        with self.assertRaises(NSyntaxError):
            lex('#include "a>')
        with self.assertRaises(NSyntaxError):
            lex('#include <a"')


class TestIncludedFile(unittest.TestCase):
    def test_eq(self):
        self.assertEqual(yan.IncludedFile(True, 'a.h'),
                         yan.IncludedFile(True, 'a.h'))


class TestParser(unittest.TestCase):
    def checkDecl(self, source, expected_result=None, parse_func=parse):
        if expected_result is None:
            expected_result = source
        tree = parse_func(source)
        result = str(tree).rstrip('\n')
        self.assertEqual(result, expected_result)

    def checkExpr(self, source, expected_result=None):
        self.checkDecl(source, expected_result, parse_expr)

    def checkStatement(self, source, expected_result=None):
        self.checkDecl(source, expected_result, parse_statement)

    def test_string_concatenation(self):
        self.checkExpr('"hello" "world" "!"')

    def test_sizeof(self):
        self.checkExpr('sizeof(int)')
        self.checkExpr('sizeof(123)')
        self.checkExpr('sizeof 123')

    def test_cast(self):
        self.checkExpr('(int)54.9')
        self.checkExpr('(char **)17')

    def test_call(self):
        self.checkExpr('a()()()')

    def test_binary_operation(self):
        self.checkExpr('1 + 2', '(1 + 2)')
        self.checkExpr('1 + 2 * 3', '(1 + (2 * 3))')
        self.checkExpr('1 * 2 + 3', '((1 * 2) + 3)')
        self.checkExpr('1 + 2 + 3', '((1 + 2) + 3)')

    def test_assign(self):
        self.checkStatement('a = 2;')
        self.checkStatement('a += 2;')

    def test_dot_operation(self):
        self.checkExpr('a.b')
        self.checkExpr('a->b')
        self.checkExpr('a->b.c')
        with self.assertRaises(NSyntaxError):
            parse_expr('a.""')
        with self.assertRaises(NSyntaxError):
            parse_expr('a->""')

    def test_precedence(self):
        self.checkExpr('1 || 2 && 3 | 4 ^ 5 & 6 == '
                       '7 > 8 >> 9 + 10 % a.b',
                       '(1 || (2 && (3 | (4 ^ (5 & (6 == '
                       '(7 > (8 >> (9 + (10 % a.b))))))))))')

    def test_statement(self):
        self.checkStatement('0;')
        self.checkStatement('1 + 2;', '(1 + 2);')

    def test_array(self):
        self.checkDecl('int b[4];')
        self.checkDecl('int b[1][2];', 'int b[1][2];')
        self.checkDecl('void f(int b[]);')

    def test_decl(self):
        self.checkDecl('long unsigned register int b, c;')
        self.checkDecl('const volatile int b, c = 1;')
        self.checkDecl('int **b, *c = 1;',)

    def test_function_decl(self):
        self.checkDecl('void f(long a);')
        self.checkDecl('void f(long);')
        self.checkDecl('void (*a)(char *, int, long *);')

    def test_function_def(self):
        self.checkDecl('void f(long a) {}',
                       'void f(long a)\n{\n}')
        self.checkDecl('void f(long) {}', 'void f(long)\n{\n}')

        self.checkDecl('int main()'
                       '{'
                       'if (a < b) a;'
                       '}',
                       'int main()\n'
                       '{\n'
                       'if ((a < b))\n'
                       'a;\n'
                       '}')

        self.checkDecl('void f(short b)\n'
                       '{\n'
                       'char *a, c;\n'
                       'int c;\n'
                       '\n'
                       '(7 += a);\n'
                       '}')

        self.checkDecl('char *strdup(const char *);')
        self.checkDecl('char *strdup(const char *s);')

    def test_function_pointer(self):
        self.checkDecl('int (*getFunc())(int, int (*)(long));')
        self.checkDecl('int (*getFunc())(int, int (*)(long)) {}',
                       'int (*getFunc())(int, int (*)(long))\n'
                       '{\n'
                       '}')

    def test_unknown_header(self):
        self.checkDecl('#include <_unknown_header_ieuaeiuaeiua_>\n'
                       'int n;',
                       'int n;')

    def test_size_t(self):
        with self.assertRaises(NSyntaxError):
            parse('size_t n;')
        self.checkDecl('#include <stdlib.h>\n'
                       'size_t n;',
                       'size_t n;')
        self.checkDecl('#include <stddef.h>\n'
                       'size_t n;',
                       'size_t n;')
        self.checkDecl('#include <unistd.h>\n'
                       'size_t n;',
                       'size_t n;')

    def test_comment(self):
        with self.assertRaises(NSyntaxError):
            parse('// no C++ comment')
        self.checkDecl('/* This is a coment */', '')
        self.checkDecl('int /* hello */ n;', 'int n;')

    def test_print_char(self):
        self.checkDecl('#include <unistd.h>\n'
                       ''
                       'void print_char(char c)'
                       '{'
                       '  write(STDOUT_FILENO, &c, 1);'
                       '}',
                       'void print_char(char c)\n'
                       '{\n'
                       'write(STDOUT_FILENO, (&c), 1);\n'
                       '}')

    def test_struct(self):
        self.checkDecl('struct s;')
        self.checkDecl('struct s s;')
        self.checkDecl('union s;')

    def test_struct_compound(self):
        self.checkDecl('struct'
                       '{'
                       'int a;'
                       '};',
                       'struct\n'
                       '{\n'
                       'int a;\n'
                       '};')

        self.checkDecl('struct s'
                       '{'
                       'int a;'
                       '};',
                       'struct s\n'
                       '{\n'
                       'int a;\n'
                       '};')

        with self.assertRaises(NSyntaxError):
            parse('struct s {a + a;}')

    def test_compound_literal(self):
        self.checkExpr('(int *){23, 45, 67}')
        self.checkExpr('(int[]){23, 45, 67}')
        self.checkExpr('(struct s *[12]){23, 45, 67}')

    def test_initializer_list(self):
        self.checkDecl('int a[] = {2, 3, 4};')

    def test_designated_initializer(self):
        self.checkDecl('struct a a = {.a = 0, .b = 1};')
        self.checkDecl('int a[] = {[0] = 0, [1] = 1};')

    def test_enum(self):
        # TODO
        # self.checkDecl('enum s;')
        pass

    def test_typedef(self):
        self.checkDecl('typedef int a;\n\n'
                       'typedef int a;')

        with self.assertRaises(NSyntaxError):
            parse('a b;')

        self.checkDecl('typedef int (*a);\n\n'
                       'typedef a b;\n\n'
                       'b c;')
        self.checkDecl('typedef int a[32];')
        self.checkDecl('typedef int (*a)();')

    def test_typedef_struct(self):
        self.checkDecl('typedef struct s_dir\n'
                       '{\n'
                       'int n;\n'
                       'int m;\n'
                       '} t_dir;\n\n'
                       't_dir d;')

        self.checkDecl('typedef char c;\n'
                       '\n'
                       'typedef struct s_dir\n'
                       '{\n'
                       'void (*a)(c);\n'
                       '} t_dir;\n\n'
                       't_dir d;')

    def test_if(self):
        self.checkStatement('if (a)\n'
                            'b;')

        self.checkStatement('{\n'
                            'if (a)\n'
                            'b;\n'
                            'c;\n'
                            '}')

    def test_while(self):
        self.checkStatement('while (a)\n'
                            'b;')

        self.checkStatement('{\n'
                            'while (a)\n'
                            'b;\n'
                            'c;\n'
                            '}')

    def test_selection(self):
        expr = parse_expr('1 + 1')
        plus = expr.select('binary_operation')
        assert len(plus) == 1
        assert isinstance(list(plus)[0], yan.BinaryOperationExpr)

        one = expr.select('literal')
        assert len(one) == 2
        for child in one:
            assert isinstance(child, yan.LiteralExpr)

        one = expr.select('binary_operation literal')
        assert len(one) == 2
        for child in one:
            assert isinstance(child, yan.LiteralExpr)

        with self.assertRaises(ValueError):
            parse_expr('123').select('')

        with self.assertRaises(ValueError):
            parse_expr('123').select('eiaueiuaeiua')
