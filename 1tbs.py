#!/usr/bin/env python3

import unittest
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


KEYWORDS = '''
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
'''.split()


SIGNS = '''
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
'''.split()


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
    directives = r'(^|(?<=\n))[ \t]*#[ \t]*(' + directive_list + r').*'

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


def check_directive(string, begin):
    string = string.strip()[1:].strip()
    if string.startswith('include'):
        system_include_pattern = r'^include\s+<[\w\.]+>$'
        local_include_pattern = r'^include\s+"[\w\.]+"$'
        if (re.match(system_include_pattern, string) is None and
                re.match(local_include_pattern, string) is None):
            msg = "Invalid #include directive (was {!r})".format('#' + string)
            raise SyntaxError(msg, begin)

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
        position = Position(file_name, end.index, end.line, end.column + 1)

        if kind == '__newline__':
            position = Position(file_name, mo.end(), position.line + 1)
            continue
        elif kind == 'comment' and '\n' in string:
            end_line = position.line + string.count('\n')
            end_column = len(string) - string.rindex('\n')
            end = Position(file_name, mo.end(), end_line, end_column)
            position = Position(file_name, mo.end(), end_line, end_column + 1)

        if kind == '__skip__':
            pass
        elif kind == '__mismatch__':
            raise SyntaxError("{!r} unexpected".format(string), begin)
        else:
            if kind == 'directive':
                check_directive(string, begin)
            if kind == 'integer_hex':
                kind = 'integer'
            elif kind == 'float_a' or kind == 'float_b':
                kind = 'float'
            elif kind == 'identifier' and string in KEYWORDS:
                kind = 'keyword'
            yield Token(kind, string, begin, end)


def lex(string, file_name='<unknown file>'):
    l = []
    for token in lex_token(string, file_name):
        l.append(token)
    return l


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
        self.assertTokenEqual('\n  #  include "a" \n ',
                              'directive', '  #  include "a" ')
        with self.assertRaises(SyntaxError):
            lex('#include "a>')
        with self.assertRaises(SyntaxError):
            lex('#include <a"')


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


def makeAbstractBracketExpr(class_name, signs, name):
    """
    Returns a new class
    """
    def __init__(self, left_bracket, right_bracket):
        assert left_bracket.kind == 'sign'
        assert left_bracket.string == signs[0]
        assert right_bracket.kind == 'sign'
        assert right_bracket.string == signs[1]
        self._left_bracket = left_bracket
        self._right_bracket = right_bracket

    def right_bracket(self):
        return self._right_bracket

    def left_bracket(self):
        return self._left_bracket

    return type(class_name, (object,), {
        '__init__': __init__,
        'left_' + name: property(left_bracket),
        'right_' + name: property(right_bracket),
    })


AbstractParenExpr = makeAbstractBracketExpr('AbstractParenExpr',
                                            '()', 'paren')

AbstractBracketExpr = makeAbstractBracketExpr('AbstractBracketExpr',
                                              '[]', 'bracket')

AbstractBraceExpr = makeAbstractBracketExpr('AbstractBraceExpr',
                                            '{}', 'brace')


class ParameterListExpr(Expr, AbstractParenExpr):
    def __init__(self, left_paren, parameters, right_paren):
        AbstractParenExpr.__init__(self, left_paren, right_paren)
        Expr.__init__(self, parameters)
        assert isinstance(parameters, list)
        self.parameters = parameters

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
        return '{} {}'.format(specifiers, self.declarator)


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


class CompoundExpr(Expr, AbstractBraceExpr):
    """
    A compound expression.
    Starts with a '{' and ends with a '}'
    """

    def __init__(self, left_brace, declarations, statements, right_brace):
        AbstractBraceExpr.__init__(self, left_brace, right_brace)
        assert isinstance(declarations, list)
        assert isinstance(statements, list)
        Expr.__init__(self, declarations + statements)
        self.declarations = declarations
        self.statements = statements

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
        if len(self.statements) > 0:
            s += ''.join(str(s) for s in self.statements)
            s += '\n'
        s += '}\n'
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


class TypeExpr(Expr):
    def __init__(self, specifiers_tokens, declarator):
        Expr.__init__(self, [] if declarator is None else [declarator])
        self.specifiers_tokens = specifiers_tokens
        self.declarator = declarator

    @property
    def tokens(self):
        tokens = self.specifiers_tokens[:]
        if self.declarator is not None:
            tokens += self.declarator.tokens
        return tokens

    def __str__(self):
        s = ' '.join(t.string for t in self.specifiers_tokens)
        if self.declarator is not None:
            s += ' ' + str(self.declarator)
        return s


class CastExpr(Expr, AbstractParenExpr):
    def __init__(self, left_paren, type_name, right_paren, expression):
        AbstractParenExpr.__init__(self, left_paren, right_paren)
        Expr.__init__(self, [type_name, expression])
        self.type_name = type_name
        self.expression = expression

    @property
    def tokens(self):
        tokens = [self.left_paren]
        tokens += self.type_name.tokens
        tokens += [self.rigth_paren]
        tokens += self.expression.tokens
        return tokens

    def __str__(self):
        s = '(' + str(self.type_name) + ') '
        s += str(self.expression)
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
        return '{} {};\n'.format(specifiers, declarators)


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
        op = self.operator.string
        s = '{} {} {}'.format(self.left, op, self.right)
        if op in '='.split():
            return s
        return '(' + s + ')'


class CallExpr(Expr, AbstractParenExpr):
    def __init__(self, expression, left_paren, arguments, right_paren):
        AbstractParenExpr.__init__(self, left_paren, right_paren)
        Expr.__init__(self, [expression] + arguments)
        self.expression = expression
        self.arguments = arguments

    @property
    def tokens(self):
        args_tokens = []
        for arg in self.arguments:
            args_tokens += arg.tokens
        return (self.expression.tokens +
                [self.left_paren] + args_tokens + [self.right_paren])

    def __str__(self):
        s = str(self.expression)
        s += '(' + ', '.join(str(a) for a in self.arguments) + ')'
        return s


class SizeofExpr(Expr):
    def __init__(self, sizeof, left_paren, expression, right_paren):
        assert sizeof.string == 'sizeof'
        if left_paren is not None:
            assert left_paren.string == '('
            assert right_paren.string == ')'
        Expr.__init__(self, [expression])
        self.sizeof = sizeof
        self.left_paren = left_paren
        self.expression = expression
        self.right_paren = right_paren

    @property
    def tokens(self):
        tokens = [self.sizeof]
        if self.left_paren is not None:
            tokens.append(self.left_paren)
        tokens += self.expression.tokens
        if self.right_paren is not None:
            tokens.append(self.right_paren)
        return tokens

    def __str__(self):
        s = 'sizeof'
        if self.left_paren is not None:
            s += '('
        s += str(self.expression)
        if self.right_paren is not None:
            s += ')'
        return s


class SubscriptExpr(Expr, AbstractBracketExpr):
    def __init__(self, expression, left_bracket, index, right_bracket):
        AbstractBracketExpr.__init__(self, left_bracket, right_bracket)
        Expr.__init__(self, [expression, index])
        self.expression = expression
        self.index = index

    @property
    def tokens(self):
        return (self.expression.tokens +
                [self.left_bracket] +
                self.index.tokens +
                [self.right_bracket])

    def __str__(self):
        return str(self.expression) + '[' + str(self.index) + ']'


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
            return '({}{})'.format(self.right, self.operator.string)
        return '({}{})'.format(self.operator.string, self.right)


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


class WhileExpr(Expr, AbstractParenExpr):
    def __init__(self, while_token, left_paren, expression, right_paren,
                 statement):
        AbstractParenExpr.__init__(self, left_paren, right_paren)
        Expr.__init__(self, [expression, statement])
        self.while_token = while_token
        self.expression = expression
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


class IfExpr(Expr, AbstractParenExpr):
    def __init__(self, if_token, left_paren, expression, right_paren,
                 statement, else_token, else_statement):
        AbstractParenExpr.__init__(self, left_paren, right_paren)
        assert if_token.string == 'if'
        children = [expression, statement]
        if else_statement is not None:
            children.append(else_statement)
        Expr.__init__(self, children)
        self.if_token = if_token
        self.expression = expression
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
        assert keyword in 'if while do'.split();
        Expr.__init__(self, [expression])
        self.keyword = keyword
        self.expression = expression
        self.semicolon = semicolon

    @property
    def tokens(self):
        return [self.keyword] + self.expression.tokens + [self.semicolon]

    def __str__(self):
        return self.keyword.string + ' ' + str(self.expression) + ';'


class ParenExpr(Expr, AbstractParenExpr):
    def __init__(self, left_paren, expression, right_paren):
        AbstractParenExpr.__init__(left_paren, expression, right_paren)
        Expr.__init__(self, [expression])
        self.expression = expression

    @property
    def tokens(self):
        return [self.left_paren] + self.expression.tokens + [self.right_paren]

    def __str__(self):
        return '(' + str(self.expression) + ')'


class TokenReader:
    """
    An utility to read a list of tokens.
    """

    def __init__(self, tokens):
        self._tokens = tokens
        self._index = 0

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index):
        assert index <= len(self._tokens)
        self._index = index

    @property
    def has_more(self):
        return self.index < len(self._tokens)

    def next(self):
        assert self._index < len(self._tokens)
        token = self._tokens[self._index]
        self._index += 1
        return token

    @property
    def position(self):
        if not self.has_more:
            return self._tokens[-1].end
        return self._tokens[self._index].end


class Parser(TokenReader):
    """
    A parser for the C language.

    Grammar: http://www.lysator.liu.se/c/ANSI-C-grammar-y.html
    """

    def __init__(self, tokens):
        TokenReader.__init__(self, tokens)
        self._types = []
        self._included_headers = []

    @property
    def types(self):
        return self._types[:]

    def add_type(self, type_name):
        if type_name in self.types:
            msg = 'Redefinition of type {!r}'.format(type_name)
            self.raise_syntax_error(msg)
        self._types.append(type_name)

    def add_types(self, types):
        for type_name in types:
            self.add_type(type_name)

    def get_system_headers_types(self):
        """
        Types are part of the syntax. We must know the defined types
        to be able to parse correctly a file. But parsing a header
        of the libc full of macros is fairly complicated: since this
        program has no real preprocessor, it doesn't handle the macros.

        So here is a list of the standard types defined in the standard
        headers.
        """
        stdint = ('intmax_t int8_t int16_t int32_t int64_t '
                  'int_least8_t int_least16_t int_least32_t int_least64_t '
                  'int_fast8_t int_fast16_t int_fast32_t int_fast64_t '
                  'intptr_t ').split()
        stdint += ['u' + type for type in stdint]
        stdint = ' '.join(stdint)

        stddef = 'size_t ptrdiff_t'
        strings = {
            'setjmp.h':         'jmp_buf',
            'stddef.h':         stddef,
            'stdint.h':         stdint,
            'stdio.h':          stddef + ' FILE',
            'stdlib.h':         stddef + ' div_t ldiv_t',
            'string.h':         stddef,
            'time.h':           stddef + ' clock_t time_t',
            'unistd.h':         stddef + ' ssize_t',
        }
        return {h: types.split() for h, types in strings.items()}

    def expand_header(self, header_name, system):
        if (header_name, system) in self._included_headers:
            return
        self._included_headers.append((header_name, system))
        headers_types = self.get_system_headers_types()
        if header_name in headers_types:
            self.add_types(headers_types[header_name])

    def process_include(self, directive_string):
        s = directive_string
        assert s.startswith('include')
        s = s[len('include'):].strip()
        if s.startswith('<') and s.endswith('>'):
            system = True
            name = s[1:-1]
        elif s.startswith('"') and s.endswith('"'):
            system = False
            name = s[1:-1]
        else:
            self.raise_syntax_error('Invalid #include directive')
        self.expand_header(name, system)

    def has_more_impl(self, index=-1):
        if index == -1:
            index = self.index
        if index >= len(self._tokens):
            return False
        next_token = self._tokens[index]
        if next_token.kind == 'comment' or next_token.kind == 'directive':
            return self.has_more_impl(index + 1)
        return True

    @property
    def has_more(self):
        return self.has_more_impl()

    def next(self):
        token = TokenReader.next(self)
        if token.kind == 'comment':
            return self.next()
        elif token.kind == 'directive':
            s = token.string.strip()
            assert s[0] == '#'
            # There may be several spaces between the '#' and the
            # next word, we need to strip these
            s = s[1:].strip()
            if s.startswith('include'):
                self.process_include(s)
            return self.next()
        return token

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
            self.raise_syntax_error('Expected {!r}'.format(sign_list))
        return id

    def parse_identifier_token(self):
        begin = self.index
        token = self.parse_token('identifier')
        if token is None:
            return token
        if token.string in self.types:
            self.index = begin
            return None
        return token

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

    def parse_declarator_brackets(self, left):
        begin = self.index
        left_bracket = self.parse_sign('[')
        if left_bracket is not None:
            if not self.has_more:
                raise SyntaxError("Expected ']'", self.position)
            constant = self.parse_constant_expression()
            if constant is not None:
                right_bracket = self.expect_sign(']')
                return SubscriptExpr(left,
                                     left_bracket, constant, right_bracket)
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
            if parameter_list is not None:
                left = FunctionExpr(left, parameter_list)
                continue
            brackets = self.parse_declarator_brackets(left)
            if brackets is not None:
                left = brackets
                continue
            break
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
        if token.kind in 'integer float string character'.split():
            return token
        self.index = begin
        return self.parse_identifier_token()

    def parse_paren_expression(self):
        begin = self.index
        left_paren = self.parse_sign('(')
        if left_paren is None:
            return None
        expr = self.parse_expression()
        if expr is None:
            self.index = begin
            return None
        return ParenExpr(left_paren, expr, self.expect_sign(')'))

    def parse_primary_expression(self):
        token = self.parse_primary_token()
        if token is not None:
            return LiteralExpr(token)
        return self.parse_paren_expression()

    def parse_initializer(self):
        expr = self.parse_assignment_expression()
        if expr is not None:
            return expr
        left_brace = self.parse_sign('{')
        if left_brace is not None:
            self.raise_syntax_error('Initializer lists are not supported')
        return None

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

    def parse_struct_or_union_specifier(self):
        kw = self.parse_keyword('struct union'.split())
        if kw is None:
            return None

    def parse_type_specifier(self):
        """
        Returns a token or None
        """
        kw = self.parse_keyword(self.get_type_specifiers_strings())
        if kw is not None:
            return kw

        begin = self.index
        token = self.parse_token('identifier')
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

    def parse_specifier_qualifier_list(self):
        """
        Returns a list of tokens
        """
        tokens = []
        token = self.parse_type_specifier()
        if token is not None:
            tokens.append(token)
        token = self.parse_type_qualifier()
        if token is not None:
            tokens.append(token)
        if len(tokens) == 0:
            return tokens
        return tokens + self.parse_specifier_qualifier_list()

    def parse_type_name(self):
        specifiers = self.parse_specifier_qualifier_list()
        if len(specifiers) == 0:
            return None
        return TypeExpr(specifiers, self.parse_declarator(abstract=True))

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
        if left is None:
            return None
        while True:
            op = self.parse_sign('[ ( ++ --'.split())
            if op is None:
                break
            if op.string == '(':
                arguments = self.parse_argument_expression_list()
                right_paren = self.expect_sign(')')
                return CallExpr(left, op, arguments, right_paren)
            elif op.string == '[':
                expr = self.parse_expression()
                right_bracket = self.expect_sign(']')
                return SubscriptExpr(left, op, expr, right_bracket)
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

    def parse_sizeof(self):
        sizeof = self.parse_keyword(['sizeof'])
        if sizeof is None:
            return None
        expr = self.parse_unary_expression()
        if expr is not None:
            return SizeofExpr(sizeof, None, expr, None)
        left_paren = self.parse_sign('(')
        if left_paren is not None:
            type_name = self.parse_type_name()
            if type_name is None:
                self.raise_syntax_error('Expected type name')
            right_paren = self.parse_expect(')')
            return SizeofExpr(sizeof, left_paren, right_paren)

    def parse_unary_expression(self):
        sizeof = self.parse_sizeof()
        if sizeof is not None:
            return sizeof
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
        expr = self.parse_unary_expression()
        if expr is not None:
            return expr
        left_paren = self.parse_sign('(')
        if left_paren is None:
            return None
        type_name = self.parse_type_name()
        if type_name is None:
            return self.raise_syntax_error('Expected type name')
        right_paren = self.expect_sign(')')
        expr = self.parse_cast_expression()
        if expr is None:
            return self.raise_syntax_error('Expected expression')
        return CastExpr(left_paren, type_name, right_paren, expr)

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

    def parse_constant_expression(self):
        return self.parse_conditional_expression()

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
                raise SyntaxError('Expected expression', self.position)
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
            raise SyntaxError('Expected compound statement', self.position)
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


def argument_to_tokens(v):
    if isinstance(v, str):
        v = lex(v)
    if not isinstance(v, list):
        raise ArgumentError('Expected a list of tokens')
    return v


def parse(v):
    """
    v: a string or a list of tokens
    """
    parser = Parser(argument_to_tokens(v))
    return parser.parse()


def parse_statement(v):
    """
    v: a string or a list of tokens
    """
    parser = Parser(argument_to_tokens(v))
    return parser.parse_statement()


def parse_expr(v):
    """
    v: a string or a list of tokens
    """
    parser = Parser(argument_to_tokens(v))
    return parser.parse_expression()


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

    def test_binary_operation(self):
        self.checkExpr('1 + 2', '(1 + 2)')
        self.checkExpr('1 + 2 * 3', '(1 + (2 * 3))')
        self.checkExpr('1 * 2 + 3', '((1 * 2) + 3)')

    def test_statement(self):
        self.checkStatement('0;')
        self.checkStatement('1 + 2;', '(1 + 2);')

    def test_array(self):
        self.checkDecl('int b[4];')
        self.checkDecl('int b[1][2];', 'int b[1][2];')

    def test_decl(self):
        self.checkDecl('long unsigned register int b, c;')
        self.checkDecl('const volatile int b, c = 1;')
        self.checkDecl('int **b, *c = 1;',
                       'int (*(*b)), (*c) = 1;')

    def test_function_decl(self):
        self.checkDecl('void f(long a);')
        self.checkDecl('void f(long);')

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

        self.checkDecl('void f(short b)'
                       '{'
                       '  char *a, c; int c; 7 += a;'
                       '}',
                       'void f(short b)\n'
                       '{\n'
                       'char (*a), c;\n'
                       'int c;\n'
                       '\n'
                       '(7 += a);\n'
                       '}')

        self.checkDecl('char *strdup(const char *);',
                       'char (*strdup(const char (*)));')
        self.checkDecl('char *strdup(const char *s);',
                       'char (*strdup(const char (*s)));')

    def test_function_pointer(self):
        self.checkDecl('int (*getFunc())(int, int (*)(long));')
        self.checkDecl('int (*getFunc())(int, int (*)(long)) {}',
                       'int (*getFunc())(int, int (*)(long))\n'
                       '{\n'
                       '}')

    def test_size_t(self):
        with self.assertRaises(SyntaxError):
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
        with self.assertRaises(SyntaxError):
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
        self.checkDecl('enum s;')
        self.checkDecl('union s;')

    def test_typedef(self):
        # TODO:
        pass


def get_argument_parser():
    descr = 'Check your C programs against the "EPITECH norm".'
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('source_files',
                        nargs='*', type=argparse.FileType('r'),
                        help='source files to check')
    parser.add_argument('--test', action='store_true',
                        help='run the old tests')
    return parser


def test():
    # TODO: Write real tests
    sources = [
        'int a(int (*)());',

        'int main(int argc, char **argv);',
        'void f(void);',
        'int (*f)(void);',
        'int (*getFunc())(int, int (*b)(long));',
        'int (*a)();',

        'main() {lol();}',

        '''
        #include <unistd.h>

        void print_char(char c)
        {
          write(STDOUT_FILENO, &c, 1);
        }

        void print_string(const char *string)
        {
          while (*string)
            {
              my_putchar(*string);
              string++;
            }
        }
        ''',
    ]
    for source in sources:
        print(source)
        print(parse(source))


def check_file(source_file):
    source = source_file.read()
    print('tokenizing...')
    tokens = lex(source)
    print('\n'.join(repr(t) for t in tokens))
    print('parsing...')
    # print('\n'.join(repr(t) for t in tokens))
    ast = parse(tokens)
    print(str(ast))


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
