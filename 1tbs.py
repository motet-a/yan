#!/usr/bin/env python3

import unittest
import collections
import re
import sys
import argparse


class Position:
    def __init__(self, file_name, index=0, line=1, column=1):
        self._file_name = file_name
        self._index = index
        self._line = line
        self._column = column

    @property
    def file_name(self):
        return self._file_name

    @property
    def index(self):
        return self._index

    @property
    def line(self):
        return self._line

    @property
    def column(self):
        return self._column

    def __add__(self, other):
        if isinstance(other, Position):
            other = other.index
        return self.index + other

    def __sub__(self, other):
        if isinstance(other, Position):
            other = other.index
        return self.index - other

    def __str__(self):
        return '{}:{}:{}'.format(
            self.file_name,
            self.line,
            self.column)


class TestPosition(unittest.TestCase):
    def test_begin_position(self):
        p = Position('abcd')
        self.assertEqual(p.file_name, 'abcd')
        self.assertEqual(p.index, 0)
        self.assertEqual(p.line, 1)
        self.assertEqual(p.column, 1)
        self.assertEqual(str(p), 'abcd:1:1')

    def test_position(self):
        p = Position('abcd', 2, 3, 4)
        self.assertEqual(p.index, 2)
        self.assertEqual(p.line, 3)
        self.assertEqual(p.column, 4)
        self.assertEqual(str(p), 'abcd:3:4')


TOKEN_KINDS = [
    'identifier',
    'integer', 'float', 'string', 'character',
    'sign',
    'keyword',
    'comment',
    'directive',
]


class Token:
    def __init__(self, kind, string, begin, end):
        assert isinstance(kind, str)
        assert kind in TOKEN_KINDS
        assert isinstance(string, str)
        assert isinstance(begin, Position)
        assert isinstance(end, Position)
        assert end - begin == len(string)
        self._string = string
        self._kind = kind
        self._begin = begin
        self._end = end

    @property
    def kind(self):
        return self._kind

    @property
    def string(self):
        return self._string

    @property
    def begin(self):
        return self._begin

    @property
    def end(self):
        return self._end

    def __str__(self):
        return self.string

    def __repr__(self):
        return '<Token kind={}, string={!r}, begin={}, end={}>'.format(
            self.kind, self.string, self.begin, self.end,
        )


class OTBSWarning:
    def __init__(self, message, position):
        assert isinstance(message, str)
        assert isinstance(position, Position)
        self.message = message
        self.position = position

    def __str__(self):
        return "{}: {}".format(self.position, self.message)


class SyntaxError(Exception, OTBSWarning):
    def __init__(self, message, position):
        Exception.__init__(self, message)
        OTBSWarning.__init__(self, message, position)
        self.message = message
        self.position = position

    def __str__(self):
        return OTBSWarning.__str__(self)


def raise_expected_string(expected_string, position):
    raise SyntaxError("Expected '{}'".format(expected_string), position)


KEYWORDS = """
auto
break
case
char
const
continue
default
do
double
else
enum
extern
float
for
goto
if
inline
int
long
register
restrict
return
short
signed
sizeof
static
struct
switch
typedef
union
unsigned
void
volatile
while
""".split()


SIGNS = """
...
>>= <<=
+= -= *= /= %=
&= ^= |=
>> <<
++ --
+ - * / %
->
&& ||
<= >=
== !=
;
{ }
,
:
=
( )
[ ]
< >
.
& ^ |
!
~
?
""".split()


def get_lexer_spec():
    """
    Returns an array of regexps which represents the "grammar"
    of the lexer.

    This is quite unintelligible.
    """

    int_suffix = r'[uUlL]*'
    float_suffix = r'[fFlL]?'
    e_suffix = '[Ee][+-]?\d+'

    hex_digit = r'[a-fA-F0-9]'
    hex_digits = hex_digit + '+'

    signs = ''
    for sign in SIGNS:
        if signs != '':
            signs += '|'
        if sign in ['...', '||', '++', '--']:
            sign = ''.join('\\' + c for c in sign)
        elif sign[0] in '+()[]^|*.?':
            sign = '\\' + sign
        signs += sign
    signs = '(' + signs + ')'

    directive_list = ('include|define|undef|'
                      'ifdef|ifndef|if|else|elif|endif|'
                      'error|pragma')
    directives = r'(^|\n)[ \t]*#[ \t]*(' + directive_list + r').*'

    return [
        ('comment',             r'/\*(\*(?!\/)|[^*])*\*/'),
        ('string',              r'"(\\.|[^\n\\"])*"'),
        ('character',           r"'(\\.|[^\n\\'])*'"),
        ('float_a',             r'\d*\.\d+' + float_suffix),
        ('float_b',             r'\d+\.\d*' + float_suffix),
        ('integer_hex',         r'0[xX]' + hex_digits + int_suffix),
        ('integer',             r'\d+' + int_suffix),
        ('identifier',          r'[_A-Za-z][_A-Za-z0-9]*'),
        ('sign',                signs),
        ('directive',           directives),
        ('__newline__',         r'\n'),
        ('__skip__',            r'[ \t]+'),
        ('__mismatch__',        r'.'),
    ]


def get_lexer_regexp():
    def get_token_regex(pair):
        return '(?P<{}>{})'.format(*pair)

    lexer_spec = get_lexer_spec()
    return '|'.join(get_token_regex(pair) for pair in lexer_spec)


def lex_token(source_string, file_name):
    position = Position(file_name)
    for mo in re.finditer(get_lexer_regexp(), source_string):
        kind = mo.lastgroup
        string = mo.group(kind)
        assert len(string) == (mo.end() - mo.start())
        begin = Position(file_name, mo.start(),
                         position.line, position.column)
        end = Position(file_name, mo.end(),
                       position.line, position.column + len(string) - 1)
        position = Position(file_name, mo.end(),
                            position.line, position.column + len(string))

        if kind == '__newline__' or (kind == 'directive' and
                                     string[0] == '\n'):
            position = Position(file_name, mo.end(), position.line + 1)
            if kind == '__newline__':
                continue

        if kind == '__skip__':
            pass
        elif kind == '__mismatch__':
            raise SyntaxError("{!r} unexpected".format(string), begin)
        else:
            if kind == 'integer_hex':
                kind = 'integer'
            if kind == 'float_a' or kind == 'float_b':
                kind = 'float'
            elif kind == 'identifier' and string in KEYWORDS:
                kind = 'keyword'
            if kind == 'directive' and string[0] == '\n':
                string = string[1:]
                begin = Position(file_name, begin.index + 1, begin.line + 1)
            yield Token(kind, string, begin, end)


def lex(string, file_name='<unknown file>'):
    l = []
    for token in lex_token(string, file_name):
        l.append(token)
    return l
    g = lex_token(string, file_name)
    return list(g)


class TestLexer(unittest.TestCase):
    def assertLexEqual(self, source, expected):
        tokens = lex(source)
        self.assertEqual(''.join(repr(t) for t in tokens), expected)

    def assertTokenEqual(self, source, kind, string):
        assert kind in TOKEN_KINDS
        tokens = lex(source)
        if len(tokens) != 1:
            print(tokens)
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
        with self.assertRaises(SyntaxError):
            lex('"abc')
        with self.assertRaises(SyntaxError):
            lex('"\n"')
        with self.assertRaises(SyntaxError):
            lex('"')

    def test_string_escape(self):
        self.assertTokenEqual(r'"\0"', 'string', r'"\0"')
        self.assertTokenEqual(r'"\n"', 'string', r'"\n"')
        with self.assertRaises(SyntaxError):
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

    def test_directive(self):
        self.assertTokenEqual('#ifdef a', 'directive', '#ifdef a')
        self.assertTokenEqual('\n#ifdef a', 'directive', '#ifdef a')
        self.assertTokenEqual('\n  #  include <a> \n ',
                              'directive', '  #  include <a> ')


class Expr:
    def __init__(self, children):
        self._children = children
        for child in children:
            assert isinstance(child, Expr)

    @property
    def children(self):
        return self._children

    @property
    def tokens(self):
        if len(self) == 0:
            raise Exception('Not implemented (this node is a leaf)')
        tokens = []
        for child in self.children:
            for t in child.tokens:
                assert isinstance(t, Token)
            tokens += child.tokens
        return tokens

    @property
    def first_token(self):
        return self.tokens[0]

    @property
    def last_token(self):
        return self.tokens[-1]

    def __len__(self):
        return len(self.children)

    def __str__(self):
        raise Exception('Not implemented')

    def __repr__(self):
        class_name = self.__class__.__name__
        s = '<{} children={}>'.format(class_name, self.children)
        return s


class ParameterListExpr(Expr):
    def __init__(self, left_paren, parameters, right_paren):
        Expr.__init__(self, parameters)
        assert left_paren.string == '('
        assert isinstance(parameters, list)
        assert right_paren.string == ')'
        self.left_paren = left_paren
        self.parameters = parameters
        self.right_paren = right_paren

    @property
    def tokens(self):
        parameters_tokens = [p.tokens for p in self.parameters]
        return self.left_paren + parameters_tokens + self.right_paren

    def __str__(self):
        return '(' + ', '.join(str(p) for p in self.parameters) + ')'


class ParameterExpr(Expr):
    def __init__(self, specifiers_tokens, declarator=None):
        Expr.__init__(self, [] if declarator is None else [declarator])
        assert isinstance(specifiers_tokens, list)
        if declarator is None:
            assert len(specifiers_tokens) > 0
        self.specifiers_tokens = specifiers_tokens
        self.declarator = declarator

    @property
    def tokens(self):
        return self.specifiers_tokens + self.declarator.tokens

    def __str__(self):
        specifiers = ' '.join(t.string for t in self.specifiers_tokens)
        if self.declarator is None:
            return specifiers
        return "{} {}".format(specifiers, self.declarator)


class StatementExpr(Expr):
    def __init__(self, expression, semicolon):
        if expression is not None:
            assert isinstance(expression, Expr)
        assert isinstance(semicolon, Token)
        Expr.__init__(self, [] if expression is None else [expression])
        self.expression = expression
        self.semicolon = semicolon

    @property
    def tokens(self):
        t = []
        if self.expression is not None:
            t += self.expression.tokens
        return t + [self.semicolon]

    def __str__(self):
        if self.expression is None:
            return ';'
        return str(self.expression) + ';'


class FunctionExpr(Expr):
    """
    A function
    """

    def __init__(self, declarator, parameters):
        assert isinstance(declarator, Expr)
        assert isinstance(parameters, ParameterListExpr)
        Expr.__init__(self, [declarator, parameters])
        self.declarator = declarator
        self.parameters = parameters

    @property
    def tokens(self):
        return (self.declarator.tokens +
                self.parameters.tokens)

    def __str__(self):
        return '{}{}'.format(self.declarator, self.parameters)


class CompoundExpr(Expr):
    """
    A compound expression.
    Starts with a '{' and ends with a '}'
    """

    def __init__(self, left_brace, declarations, statements, right_brace):
        assert isinstance(left_brace, Token)
        assert isinstance(declarations, list)
        assert isinstance(statements, list)
        assert isinstance(right_brace, Token)
        Expr.__init__(self, declarations + statements)
        self.left_brace = left_brace
        self.declarations = declarations
        self.statements = statements
        self.right_brace = right_brace

    @property
    def tokens(self):
        return ([self.left_brace] +
                self.declarations.tokens +
                self.statements.tokens +
                [self.rigth_brace])

    def __str__(self):
        s = '{\n'
        if len(self.declarations) > 0:
            s += ''.join(str(d) for d in self.declarations)
            s += '\n'
        s += '\n'.join(str(s) for s in self.statements)
        s += '\n}\n'
        return s


class FunctionDefinitionExpr(Expr):
    """
    A function definition
    """

    def __init__(self, specifiers_tokens, declarator, compound):
        """
        Warning: if the function returns a pointer, `declarator`
        is a PointerExpr.
        """
        assert isinstance(specifiers_tokens, list)
        assert isinstance(compound, CompoundExpr)

        Expr.__init__(self, [declarator, compound])
        self.specifiers_tokens = specifiers_tokens
        self.declarator = declarator
        self.compound = compound

        assert isinstance(self.function, FunctionExpr)

    @property
    def function(self):
        decl = self.declarator
        while not isinstance(decl, FunctionExpr):
            if isinstance(decl, PointerExpr):
                decl = decl.right
            else:
                raise Exception()
        return decl

    @property
    def parameters(self):
        return self.function.parameters

    @property
    def tokens(self):
        return (self.specifiers_tokens +
                self.declarator.tokens +
                self.compound)

    def __str__(self):
        s = ' '.join(t.string for t in self.specifiers_tokens)
        if len(s) > 0:
            s += ' '
        s += str(self.declarator)
        s += '\n' + str(self.compound)
        return s


class DeclarationExpr(Expr):
    def __init__(self, specifiers_tokens, declarators, semicolon_token):
        Expr.__init__(self, declarators)
        assert isinstance(specifiers_tokens, list)
        assert isinstance(declarators, list)
        self.specifiers_tokens = specifiers_tokens
        self.declarators = declarators
        self.semicolon_token = semicolon_token

    @property
    def tokens(self):
        decl_tokens = []
        for declarator in self.declarators:
            decl_tokens += declarator.tokens
        semicolon = self.semicolon_token
        return (self.specifiers_tokens + decl_tokens + [semicolon])

    def __str__(self):
        specifiers = ' '.join(t.string for t in self.specifiers_tokens)
        declarators = ', '.join(str(d) for d in self.declarators)
        return "{} {};\n".format(specifiers, declarators)


class BinaryOperationExpr(Expr):
    def __init__(self, left, operator, right):
        Expr.__init__(self, [left, right])
        self.left = left
        self.operator = operator
        self.right = right

    @property
    def tokens(self):
        return self.left.tokens + [self.operator] + self.left.tokens

    def __str__(self):
        return "({} {} {})".format(self.left,
                                   self.operator.string,
                                   self.right)


class CallExpr(Expr):
    def __init__(self, expression, left_paren, arguments, right_paren):
        Expr.__init__(self, [expression] + arguments)
        self.expression = expression
        self.left_paren = left_paren
        self.arguments = arguments
        self.right_paren = right_paren

    @property
    def tokens(self):
        return (self.expression.tokens +
                [self.left_paren] + self.arguments + [self.right_paren])

    def __str__(self):
        s = str(self.expression)
        s += '(' + ', '.join(str(a) for a in self.arguments) + ')'
        return s


class PointerExpr(Expr):
    def __init__(self, star, right, type_qualifiers):
        assert(star.string == '*')
        Expr.__init__(self, [] if right is None else [right])
        self.star = star
        self.right = right
        self.type_qualifiers = type_qualifiers

    @property
    def tokens(self):
        tokens = [self.star]
        if self.right is not None:
            tokens += self.right.tokens
        return tokens

    def __str__(self):
        s = '(*'
        s += ' '.join(q.string for q in self.type_qualifiers)
        if len(self.type_qualifiers) > 0:
            s += ' '
        if self.right is not None:
            s += str(self.right)
        s += ')'
        return s


class UnaryOperationExpr(Expr):
    def __init__(self, operator, right, postfix=False):
        if postfix:
            assert operator.string in '++ --'.split()
        assert isinstance(operator, Token)
        assert isinstance(right, Expr)
        Expr.__init__(self, [right])
        self.operator = operator
        self.right = right
        self.postfix = postfix

    @property
    def tokens(self):
        return [self.operator] + self.right.tokens

    def __str__(self):
        if self.postfix:
            return "({}{})".format(self.right, self.operator.string)
        return "({}{})".format(self.operator.string, self.right)


class LiteralExpr(Expr):
    def __init__(self, token):
        Expr.__init__(self, [])
        literals = 'identifier integer float string character'.split()
        assert token.kind in literals
        self.token = token

    @property
    def kind(self):
        return self.token.kind

    @property
    def tokens(self):
        return [self.token]

    @property
    def string(self):
        return self.token.string

    def __str__(self):
        return self.token.string

    def __repr__(self):
        return '<LiteralExpr kind={} token={!r}>'.format(
            self.kind, self.string)


class WhileExpr(Expr):
    def __init__(self, while_token, left_paren, expression, right_paren,
                 statement):
        Expr.__init__(self, [expression, statement])
        self.while_token = while_token
        self.left_paren = left_paren
        self.expression = expression
        self.right_paren = right_paren
        self.statement = statement

    @property
    def tokens(self):
        tokens = [self.while_token, self.left_paren]
        tokens += self.expression.tokens
        tokens += [right_paren]
        tokens += self.statement.tokens
        return tokens

    def __str__(self):
        s = 'while (' + str(self.expression) + ')\n'
        s += str(self.statement)
        return s


class IfExpr(Expr):
    def __init__(self, if_token, left_paren, expression, right_paren,
                 statement, else_token, else_statement):
        children = [expression, statement]
        if else_statement is not None:
            children.append(else_statement)
        Expr.__init__(self, children)
        self.if_token = if_token
        self.left_paren = left_paren
        self.expression = expression
        self.right_paren = right_paren
        self.statement = statement
        self.else_token = else_token
        self.else_statement = else_statement

    @property
    def tokens(self):
        tokens = [self.if_token, self.left_paren]
        tokens += self.expression.tokens
        tokens += [right_paren]
        tokens += self.statement.tokens
        if self.else_token is not None:
            tokens.append(self.else_token)
            tokens += self.else_statement.tokens
        return tokens

    def __str__(self):
        s = 'if (' + str(self.expression) + ')\n'
        s += str(self.statement)
        if self.else_token is not None:
            s += '\nelse\n'
            s += str(self.else_statement)
        return s


class TranslationUnitExpr(Expr):
    def __init__(self, children):
        Expr.__init__(self, children)

    def __str__(self):
        return '\n'.join(str(c) for c in self.children)


class JumpStatementExpr(Expr):
    def __init__(self, keyword, expression, semicolon):
        Expr.__init__(self, [expression])
        self.keyword = keyword
        self.expression = expression
        self.semicolon = semicolon

    @property
    def tokens(self):
        return [self.keyword] + self.expression.tokens + [self.semicolon]

    def __str__(self):
        return self.keyword.string + ' ' + str(self.expression) + ';'


class TokenReader:
    def __init__(self, tokens):
        self._tokens = tokens
        self._index = 0

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index):
        assert index < len(self._tokens)
        self._index = index

    @property
    def has_more(self):
        return self.index < len(self._tokens)

    def next(self):
        assert self.has_more
        token = self._tokens[self._index]
        self._index += 1
        return token

    @property
    def position(self):
        if not self.has_more:
            return self._tokens[-1].end
        return self._tokens[self._index].end


#
# from http://www.lysator.liu.se/c/ANSI-C-grammar-y.html
#


class Parser(TokenReader):
    def __init__(self, tokens):
        TokenReader.__init__(self, tokens)
        self._types = []

    @property
    def types(self):
        return self._types[:]

    def add_type(self, type_name):
        self._types.append(type_name)

    def raise_syntax_error(self, message):
        raise SyntaxError(message, self.position)

    def parse_token(self, kind, string_list=None):
        if isinstance(string_list, str):
            string_list = [string_list]
        if not self.has_more:
            return None
        begin = self.index
        t = self.next()
        if t.kind == kind:
            if (string_list is not None) and t.string in string_list:
                return t
            elif string_list is None:
                return t
        self.index = begin

    def parse_keyword(self, keyword_list=None):
        for kw in keyword_list:
            assert kw in KEYWORDS
        return self.parse_token('keyword', keyword_list)

    def parse_sign(self, sign_list=None):
        if sign_list is None:
            sign_list = []
        for sign in sign_list:
            assert sign in SIGNS
        return self.parse_token('sign', sign_list)

    def expect_sign(self, sign_list):
        id = self.parse_sign(sign_list)
        if id is None:
            raise SyntaxError('Expected {!r}'.format(sign_list),
                              self.position)
        return id

    def parse_identifier_token(self):
        return self.parse_token('identifier')

    def expect_identifier_token(self):
        id = self.parse_identifier_token()
        if id is None:
            raise SyntaxError('Expected identifier', self.position)

    def parse_identifier(self):
        token = self.parse_identifier_token()
        if token is None:
            return None
        return LiteralExpr(token)

    def parse_parameter_declaration(self):
        specifiers = self.parse_declaration_specifiers()
        if len(specifiers) == 0:
            return None
        declarator = self.parse_declarator(False)
        if declarator is None:
            declarator = self.parse_declarator(True)
        # declarator can be None
        return ParameterExpr(specifiers, declarator)

    def parse_parameter_type_list(self):
        left_paren = self.parse_sign('(')
        if left_paren is None:
            return None
        params = []
        while self.has_more:
            if len(params) > 0 and self.parse_sign(',') is None:
                break
            param = self.parse_parameter_declaration()
            if param is None:
                if len(params) > 0:
                    raise SyntaxError("Expected parameter after ','",
                                      self.position)
                break
            params.append(param)
        right_paren = self.expect_sign(')')
        return ParameterListExpr(left_paren, params, right_paren)

    def parse_declarator_parens(self, abstract=False):
        begin = self.index
        left_paren = self.parse_sign('(')
        if left_paren is not None:
            if not self.has_more:
                raise SyntaxError("Expected ')'", self.position)
            decl = self.parse_declarator(abstract)
            if decl is not None:
                right_paren = self.expect_sign(')')
                return decl
            self.index = begin
        return None

    def parse_direct_abstract_declarator(self):
        left = self.parse_declarator_parens(True)
        if left is None:
            return None
        while True:
            decl = self.parse_declarator_parens(True)
            if decl is None:
                params = self.parse_parameter_type_list()
                if params is None:
                    return left
                return FunctionExpr(left, params)
            left = decl

    def parse_direct_declarator(self, abstract=False):
        """
        Returns a declarator or None
        """
        if abstract:
            return self.parse_direct_abstract_declarator()

        left = self.parse_declarator_parens()
        if left is None:
            left = self.parse_identifier()
        if left is None:
            return None

        while self.has_more:
            parameter_list = self.parse_parameter_type_list()
            if parameter_list is None:
                break
            else:
                left = FunctionExpr(left, parameter_list)
        return left

    def parse_declarator(self, abstract=False):
        """
        Returns a declarator or None
        """
        begin = self.index
        star = self.parse_sign('*')
        if star is None:
            return self.parse_direct_declarator(abstract)
        type_qualifiers = self.parse_type_qualifier_list()
        right = self.parse_declarator(abstract)
        if right is None and not abstract:
            self.index = begin
            return None
        return PointerExpr(star, right, type_qualifiers)

    def parse_primary_token(self):
        if not self.has_more:
            return None
        begin = self.index
        token = self.next()
        if token.kind in 'identifier integer float string character'.split():
            return token
        self.index = begin
        return None

    def parse_paren_expression(self):
        left_paren = self.parse_sign('(')
        if left_paren is None:
            return None
        expr = self.parse_expression()
        return ParenExpr(left_paren, expr, expect_sign(')'))

    def parse_primary_expression(self):
        token = self.parse_primary_token()
        if token is not None:
            return LiteralExpr(token)
        return self.parse_paren_expression()

    def parse_initializer_list(self):
        pass

    def parse_initializer(self):
        return self.parse_assignment_expression()

    def parse_init_declarator(self):
        """
        Returns a declarator or None
        """
        declarator = self.parse_declarator()
        eq = self.parse_sign('=')
        if declarator is not None and eq is not None:
            initializer = self.parse_initializer()
            if initializer is None:
                raise SyntaxError('Initializer expected', self.position)
            return BinaryOperationExpr(declarator, eq, initializer)
        return declarator

    def parse_init_declarator_list(self):
        declarators = []
        while self.has_more:
            declarator = self.parse_init_declarator()
            if declarator is None:
                break
            declarators.append(declarator)
            comma = self.parse_sign(',')
            if comma is None:
                break
        if len(declarators) == 0:
            raise SyntaxError('Declarator expected', self.position)
        return declarators

    def parse_storage_class_specifier(self):
        """
        Returns a token or None
        """
        kw_strings = 'typedef extern static auto register'.split()
        return self.parse_keyword(kw_strings)

    def get_type_specifiers_strings(self):
        return 'void char short int long float double signed unsigned'.split()

    def parse_type_specifier(self):
        """
        Returns a token or None
        """
        kw = self.parse_keyword(self.get_type_specifiers_strings())
        if kw is not None:
            return kw
        begin = self.index
        token = self.parse_identifier_token()
        if token is not None and token.string in self.types:
            return token
        self.index = begin

    def parse_type_qualifier(self):
        """
        Returns a token or None
        """
        kw_strings = ('const volatile '.split())
        return self.parse_keyword(kw_strings)

    def parse_type_qualifier_list(self):
        """
        Returns a list of tokens
        """
        qualifiers = []
        while True:
            qualifier = self.parse_type_qualifier()
            if qualifier is None:
                break
            qualifiers.append(qualifier)
        return qualifiers

    def parse_declaration_specifiers(self):
        """
        Returns a list of tokens
        """
        tokens = []
        token = self.parse_storage_class_specifier()
        if token is not None:
            tokens.append(token)
        token = self.parse_type_specifier()
        if token is not None:
            tokens.append(token)
        token = self.parse_type_qualifier()
        if token is not None:
            tokens.append(token)
        if len(tokens) == 0:
            return tokens
        return tokens + self.parse_declaration_specifiers()

    def filter_type_specifiers(self, tokens):
        return [t for t in tokens if
                t.string in self.get_type_specifiers_strings()]

    def parse_declaration(self):
        begin = self.index
        specifiers = self.parse_declaration_specifiers()
        if len(specifiers) == 0:
            return None
        declarators = self.parse_init_declarator_list()
        semicolon = self.parse_sign(';')
        if semicolon is None:
            self.index = begin
            return None
        return DeclarationExpr(specifiers, declarators, semicolon)

    def parse_declaration_list(self):
        declarations = []
        while True:
            decl = self.parse_declaration()
            if decl is None:
                break
            declarations.append(decl)
        return declarations

    def parse_argument_expression_list(self):
        arguments = []
        while True:
            if len(arguments) > 0:
                comma = self.parse_sign(',')
                if comma is None:
                    break
            argument = self.parse_assignment_expression()
            if argument is None:
                if len(arguments) == 0:
                    break
                raise SyntaxError('Expected argument', self.position)
            arguments.append(argument)
        return arguments

    def parse_postfix_expression(self):
        left = self.parse_primary_expression()
        while True:
            op = self.parse_sign('( ++ --'.split())
            if op is None:
                break
            if op.string == '(':
                arguments = self.parse_argument_expression_list()
                right_paren = self.expect_sign(')')
                return CallExpr(left, op.string, arguments, right_paren)
            elif op.string in '++ --'.split():
                return UnaryOperationExpr(op, left, postfix=True)
            else:
                raise Exception()
        return left

    def parse_unary_operator(self):
        """
        Returns a token or None
        """
        return self.parse_sign('& * + - ~ !'.split())

    def parse_unary_expression(self):
        op = self.parse_unary_operator()
        if op is None:
            op = self.parse_sign('-- ++'.split())
            if op is None:
                return self.parse_postfix_expression()
            expr = self.parse_unary_expression()
            return UnaryOperationExpr(op, expr)
        expr = self.parse_cast_expression()
        return UnaryOperationExpr(op, expr)

    def parse_binary_operation(self, operators, sub_function):
        """
        operators: a string or a list of strings
        sub_function: a function
        """

        if isinstance(operators, str):
            operators = operators.split()
        left = sub_function()
        while True:
            op = self.parse_sign(operators)
            if op is None:
                break
            left = BinaryOperationExpr(left, op, sub_function())
        return left

    def parse_cast_expression(self):
        return self.parse_unary_expression()

    def parse_multiplicative_expression(self):
        return self.parse_binary_operation('* / %',
                                           self.parse_cast_expression)

    def parse_additive_expression(self):
        f = self.parse_multiplicative_expression
        return self.parse_binary_operation('+ -', f)

    def parse_shift_expression(self):
        return self.parse_binary_operation('>> <<',
                                           self.parse_additive_expression)

    def parse_relational_expression(self):
        return self.parse_binary_operation('< > <= >=',
                                           self.parse_shift_expression)

    def parse_equality_expression(self):
        return self.parse_binary_operation('== !=',
                                           self.parse_relational_expression)

    def parse_logical_or_expression(self):
        return self.parse_equality_expression()

    def parse_conditional_expression(self):
        return self.parse_logical_or_expression()

    def parse_assignment_operator(self):
        """
        Returns an assignment operator or None
        """
        ops = '= *= /= %= += -= <<= >>= &= ^= |='.split()
        return self.parse_sign(ops)

    def parse_assignment_expression_2(self):
        begin = self.index
        unary = self.parse_unary_expression()
        if unary is None:
            return None
        op = self.parse_assignment_operator()
        if op is None:
            self.index = begin
            return None
        right = self.parse_assignment_expression()
        return BinaryOperationExpr(unary, op, right)

    def parse_assignment_expression(self):
        begin = self.index
        left = self.parse_conditional_expression()
        if left is None:
            return None
        left_end = self.index
        self.index = begin
        assign = self.parse_assignment_expression_2()
        if assign is None:
            self.index = left_end
            return left
        return assign

    def parse_expression(self):
        return self.parse_assignment_expression()

    def parse_expression_statement(self):
        semicolon = self.parse_sign(';')
        if semicolon is not None:
            return StatementExpr(None, semicolon)
        expr = self.parse_expression()
        if expr is None:
            return None
        return StatementExpr(expr, self.expect_sign(';'))

    def parse_selection_statement(self):
        switch_token = self.parse_keyword(['switch'])
        if switch_token is not None:
            raise SyntaxError("The 'switch' statement is not implemented",
                              self.position)

        if_token = self.parse_keyword(['if'])
        if if_token is None:
            return None
        left_paren = self.expect_sign('(')
        expr = self.parse_expression()
        if expr is None:
            raise SyntaxError('Expected expression', self.position)
        right_paren = self.expect_sign(')')
        statement = self.parse_statement()
        if statement is None:
            raise SyntaxError('Expected statement', self.position)

        else_statement = None
        else_token = self.parse_keyword(['else'])
        if else_token is not None:
            else_statement = self.parse_statement()
            if else_statement is None:
                raise SyntaxError("Expected statement after 'else'",
                                  self.position)
        return IfExpr(if_token, left_paren, expr, right_paren, statement,
                      else_token, else_statement)

    def parse_iteration_statement(self):
        token = self.parse_keyword('do for'.split())
        if token is not None:
            raise SyntaxError("'do' and 'for' statements are not implemented",
                              self.position)
        while_token = self.parse_keyword('while'.split())
        if while_token is None:
            return None
        left_paren = self.expect_sign('(')
        expr = self.parse_expression()
        if expr is None:
            raise SyntaxError('Expected expression', self.position)
        right_paren = self.expect_sign(')')
        statement = self.parse_statement()
        if statement is None:
            raise SyntaxError('Expected statement', self.position)
        return WhileExpr(while_token, left_paren, expr, right_paren, statement)

    def parse_jump_statement(self):
        return_token = self.parse_keyword(['return'])
        if return_token is None:
            return None
        expr = None
        semicolon = self.parse_sign(';')
        if semicolon is None:
            expr = self.parse_expression()
            if expr is None:
                raise SyntaxError('Expected expression')
            self.expect_sign(';')
        return JumpStatementExpr(return_token, expr, semicolon)

    def parse_statement(self):
        s = self.parse_compound_statement()
        if s is not None:
            return s
        s = self.parse_expression_statement()
        if s is not None:
            return s
        s = self.parse_selection_statement()
        if s is not None:
            return s
        s = self.parse_iteration_statement()
        if s is not None:
            return s
        s = self.parse_jump_statement()
        if s is not None:
            return s
        return None

    def parse_statement_list(self):
        statements = []
        while True:
            statement = self.parse_statement()
            if statement is None:
                break
            statements.append(statement)
        return statements

    def parse_compound_statement(self):
        left_brace = self.parse_sign('{')
        if left_brace is None:
            return None
        declarations = self.parse_declaration_list()
        statements = self.parse_statement_list()
        right_brace = self.expect_sign('}')
        return CompoundExpr(left_brace, declarations, statements, right_brace)

    def parse_function_definition(self):
        specifiers = self.parse_declaration_specifiers()
        declarator = self.parse_declarator()
        if declarator is None:
            if len(specifiers) == 0:
                return None
            raise SyntaxError('Expected declarator', self.position)
        compound = self.parse_compound_statement()
        if compound is None:
            raise SyntaxError('Expected coumpound statement', self.position)
        return FunctionDefinitionExpr(specifiers, declarator, compound)

    def parse_external_declaration(self):
        decl = self.parse_declaration()
        if decl is not None:
            return decl
        return self.parse_function_definition()

    def parse_translation_unit(self):
        declarations = []
        while self.has_more:
            decl = self.parse_external_declaration()
            if decl is None and self.has_more:
                pos = self.position
                raise SyntaxError('Unexpected ' + self.next().string, pos)
            if decl is not None:
                declarations.append(decl)
        return TranslationUnitExpr(declarations)

    def parse(self):
        return self.parse_translation_unit()


def remove_comments_and_directives(tokens):
    new_tokens = []
    for token in tokens:
        if token.kind not in 'comment directive'.split():
            new_tokens.append(token)
    return new_tokens


def parse(v):
    if isinstance(v, str):
        return parse(remove_comments_and_directives(lex(v)))
    assert isinstance(v, list)
    parser = Parser(v)
    return parser.parse()


def get_argument_parser():
    descr = 'Check your C programs against the "EPITECH norm".'
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('source_files',
                        nargs='*', type=argparse.FileType('r'),
                        help='source files to check')
    parser.add_argument('--test', action='store_true',
                        help='run the tests')
    return parser


def test():
    unittest.main(exit=False, argv=sys.argv[:1])

    # TODO: Write real tests
    sources = [
        'int b;',
        'void f(long a) {}',
        'void f(short b) {char *a, c; int c; 7 += a;}',
        'int main() {if (a < b) a;}',
        'char *strdup(const char *);',
        'char (*strdup)(const char *);',

        'int a(int (*)());',
        'int a(int *);',


        'long unsigned register int b, c;',
        'const volatile int b, c = 1;',
        'int **b, *c = 1;',
        'int main(int argc, char **argv);',
        'void f(void);',
        'int (*f)(void);',
        'int (*getFunc())(int, int (*b)(long));',
        'int (*a)();',

        'int (*getFunc())(int, int (*)(long));',
        'int (*getFunc())(int, int (*)(long)) {}',

        'main() {lol();}',

        '''
        #include <unistd.h>

        void my_putchar(char c)
        {
          write(STDOUT_FILENO, &c, 1);
        }

        void my_putstr(const char *string)
        {
          while (*string)
            {
              my_putchar(*string);
              string++;
            }
        }
        ''',

        '/**/',
    ]
    for source in sources:
        print(source)
        print(parse(source))


def check_file(source_file):
    source = source_file.read()
    tokens = lex(source)
    print('\n'.join(repr(t) for t in tokens))
    print(parse(source))


def main():
    argument_parser = get_argument_parser()
    args = argument_parser.parse_args()
    if args.test:
        test()
        return
    for source_file in args.source_files:
        check_file(source_file)
        source_file.close()


if __name__ == '__main__':
    main()
