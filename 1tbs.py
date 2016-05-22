#!/usr/bin/env python3

# pylint: disable=too-many-lines

"""
A C brace style checker designed for the "EPITECH norm".
"""

import argparse
import os
import re
import sys
import unittest


class Position:
    """
    Represents a position in a file.
    """

    def __init__(self, file_name, index=0, line=1, column=1):
        """
        file_name: The file path.
        index: The index relative to the begin of the file
        line: The 1-based line number
        column: The 1-based column number
        """

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


class AbstractIssue:
    def __init__(self, message, position):
        assert isinstance(message, str)
        assert isinstance(position, Position)
        self._message = message
        self._position = position

    @property
    def message(self):
        return self._message

    @property
    def position(self):
        return self._position

    def __str__(self):
        return "{}: {}".format(self.position, self.message)


class SyntaxError(Exception, AbstractIssue):
    def __init__(self, message, position):
        Exception.__init__(self, message)
        AbstractIssue.__init__(self, message, position)

    def __str__(self):
        return AbstractIssue.__str__(self)


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
->
+ - * / %
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
    # pylint: disable=bad-whitespace

    int_suffix = r'[uUlL]*'
    float_suffix = r'[fFlL]?'
    e_suffix = r'[Ee][+-]?\d+'

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
        system_include_pattern = r'^include\s+<[\w\./]+>$'
        local_include_pattern = r'^include\s+"[\w\./]+"$'
        if (re.match(system_include_pattern, string) is None and
                re.match(local_include_pattern, string) is None):
            msg = "Invalid #include directive (was {!r})".format('#' + string)
            raise SyntaxError(msg, begin)


def lex_token(source_string, file_name):
    position = Position(file_name)
    for match in re.finditer(get_lexer_regexp(), source_string):
        kind = match.lastgroup
        string = match.group(kind)
        assert len(string) == (match.end() - match.start())
        begin = Position(file_name, match.start(),
                         position.line, position.column)
        end = Position(file_name, match.end(),
                       position.line, position.column + len(string) - 1)
        position = Position(file_name, end.index, end.line, end.column + 1)

        if kind == '__newline__':
            position = Position(file_name, match.end(), position.line + 1)
            continue
        elif kind == 'comment' and '\n' in string:
            end_line = position.line + string.count('\n')
            end_column = len(string) - string.rindex('\n')
            end = Position(file_name, match.end(), end_line, end_column)
            position = Position(file_name,
                                match.end(), end_line, end_column + 1)

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
        with self.assertRaises(SyntaxError):
            lex('#include "a>')
        with self.assertRaises(SyntaxError):
            lex('#include <a"')


class Expr:
    """
    Represents a node of the syntax tree.

    An Expr is immutable.
    """

    def __init__(self, children):
        self._children = children
        for child in children:
            assert isinstance(child, Expr)

    @staticmethod
    def _split_camel_case(string):
        def find_uppercase_letter(string):
            for i, c in enumerate(string):
                if c.isupper():
                    return i
            return -1

        first = string[0].lower()
        string = first + string[1:]
        upper_index = find_uppercase_letter(string)
        if upper_index == -1:
            return [string]
        left = string[:upper_index]
        return [left] + Expr._split_camel_case(string[upper_index:])

    @staticmethod
    def _get_class_short_name(name):
        name = name[:-len('Expr')]
        l = Expr._split_camel_case(name)
        return '_'.join(l)

    @staticmethod
    def _get_expr_classes():
        import inspect
        classes = inspect.getmembers(sys.modules[__name__], inspect.isclass)
        d = {}
        for name, cls in classes:
            if name.endswith('Expr') and name != 'Expr':
                short_name = Expr._get_class_short_name(name)
                d[short_name] = cls
        d['expr'] = Expr
        return d

    def select_classes(self, cls):
        l = []
        if isinstance(self, cls):
            l.append(self)
        for child in self.children:
            l += child.select_classes(cls)
        return frozenset(l)

    @staticmethod
    def select_all(expressions, selectors_string):
        l = []
        for e in expressions:
            l += e.select(selectors_string)
        return frozenset(l)

    def select(self, selectors_string):
        """
        Selects child nodes from a "selectors" string, a bit like
        CSS selection.

        Returns a set.
        """
        selectors = selectors_string.split()
        if len(selectors) == 0:
            raise ValueError('No selector in the given string')
        selector = selectors[0]
        expr_classes = Expr._get_expr_classes()
        selected = []
        if selector in expr_classes:
            selected = self.select_classes(expr_classes[selector])
        else:
            raise ValueError('Invalid selector: {!r}'.format(selector))
        if len(selectors) > 1:
            return Expr.select_all(selected, ' '.join(selectors[1:]))
        return selected

    @property
    def children(self):
        return self._children[:]

    @property
    def tokens(self):
        tokens = []
        for child in self.children:
            for t in child.tokens:
                if not isinstance(t, Token):
                    print()
                    print(repr(child))
                    print(repr(child.tokens))
                    print()
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


class CommaListExpr(Expr):
    """
    Reprensents a list of expressions separated by commas
    """

    def __init__(self, comma_separated_children, allow_trailing=False):
        """
        comma_separated_children is a list of expressions
        separated by commas tokens.
        """

        if len(comma_separated_children) > 0 and not allow_trailing:
            assert len(comma_separated_children) % 2 == 1

        children = [d for d in comma_separated_children
                    if isinstance(d, Expr)]

        Expr.__init__(self, children)
        self._comma_separated_children = comma_separated_children

    @property
    def comma_separated_children(self):
        return self._comma_separated_children[:]

    @property
    def tokens(self):
        tokens = []
        for i in self.comma_separated_children:
            if isinstance(i, Token):
                tokens.append(i)
            else:
                tokens += i.tokens
        return tokens

    def __str__(self):
        return ', '.join(str(child) for child in self.children)


def make_abstract_bracket_expr(class_name, signs, name):
    """
    Returns a new class
    """
    def __init__(self, left_bracket, right_bracket):
        # pylint fails to analyse properly this function
        # pylint disable=protected-access

        assert isinstance(left_bracket, Token)
        assert left_bracket.kind == 'sign'
        assert left_bracket.string == signs[0]

        assert isinstance(right_bracket, Token)
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


AbstractParenExpr = make_abstract_bracket_expr('AbstractParenExpr',
                                               '()', 'paren')

AbstractBracketExpr = make_abstract_bracket_expr('AbstractBracketExpr',
                                                 '[]', 'bracket')

AbstractBraceExpr = make_abstract_bracket_expr('AbstractBraceExpr',
                                               '{}', 'brace')


class AbstractTypeSpecifierExpr(Expr):
    def __init__(self, children):
        Expr.__init__(self, children)


class TypeSpecifierExpr(AbstractTypeSpecifierExpr):
    def __init__(self, specifier_token):
        assert (specifier_token.kind == 'identifier' or
                specifier_token.kind == 'keyword')
        Expr.__init__(self, [])
        self._token = specifier_token

    @property
    def token(self):
        return self._token

    @property
    def tokens(self):
        return [self.token]

    def __str__(self):
        return self.token.string


class StructExpr(AbstractTypeSpecifierExpr):
    """
    Represents a struct or an union
    """

    def __init__(self, struct, identifier, compound):
        """
        struct: The keyword `struct` or `union`
        identifier: The name of the struct or None
        compound: The "body" of the struct or None
        """
        assert isinstance(struct, Token)
        assert (struct.string == 'struct' or
                struct.string == 'union')

        if identifier is not None:
            assert isinstance(identifier, Token)
            assert identifier.kind == 'identifier'

        if compound is not None:
            assert isinstance(compound, CompoundExpr)
            assert len(compound.statements) == 0

        children = [] if compound is None else [compound]
        AbstractTypeSpecifierExpr.__init__(self, children)

        self._struct = struct
        self._identifier = identifier
        self._compound = compound

    @property
    def kind(self):
        """
        Returns 'struct' or 'union'
        """
        return self.struct.string

    @property
    def struct(self):
        return self._struct

    @property
    def identifier(self):
        return self._identifier

    @property
    def compound(self):
        return self._compound

    @property
    def tokens(self):
        t = [self.struct]
        if self.identifier is not None:
            t.append(self.identifier)
        if self.compound is not None:
            t += self.compound.tokens
        return t

    def __str__(self):
        s = self.kind
        if self.identifier is not None:
            s += ' ' + self.identifier.string
        if self.compound is not None:
            s += '\n' + str(self.compound)
        return s


class EnumExpr(AbstractTypeSpecifierExpr):
    """
    Represents an enum
    """

    def __init__(self, enum, identifier, body):
        """
        enum: The keyword `enum`
        identifier: The name of the enum or None
        body: The "body" of the enum or None
        """
        assert isinstance(enum, Token)
        assert enum.string == 'enum'

        if identifier is not None:
            assert isinstance(identifier, Token)
            assert identifier.kind == 'identifier'

        if body is not None:
            assert isinstance(body, InitializerListExpr)

        children = [] if body is None else [body]
        AbstractTypeSpecifierExpr.__init__(self, children)

        self._enum = enum
        self._identifier = identifier
        self._body = body

    @property
    def enum(self):
        return self._enum

    @property
    def identifier(self):
        return self._identifier

    @property
    def body(self):
        return self._body

    @property
    def tokens(self):
        t = [self.enum]
        if self.identifier is not None:
            t.append(self.identifier)
        if self.body is not None:
            t += self.body.tokens
        return t

    def __str__(self):
        s = 'enum'
        if self.identifier is not None:
            s += ' ' + self.identifier.string
        if self.body is not None:
            s += '\n' + str(self.body)
        return s


class TypeExpr(Expr):
    def __init__(self, children):
        assert len(children) > 0
        Expr.__init__(self, children)

    @property
    def is_typedef(self):
        for child in self.children:
            if isinstance(child, TypeSpecifierExpr):
                if child.token.string == 'typedef':
                    return True
        return False

    def __str__(self):
        return ' '.join(str(c) for c in self.children)


class ParameterExpr(Expr):
    def __init__(self, type_expr, declarator=None):
        """
        type_expr: a TypeExpr or None, if the declarator is not None
        """
        if declarator is None or type_expr is not None:
            assert isinstance(type_expr, TypeExpr)

        self._type_expr = type_expr
        children = [type_expr] + ([] if declarator is None else [declarator])
        Expr.__init__(self, children)
        self.declarator = declarator

    @property
    def type_expr(self):
        return self._type_expr

    def __str__(self):
        specifiers = str(self.type_expr)
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
        """
        No newline at the end of the returned string because
        statements in `for` loops are usually on the same line
        """
        if self.expression is None:
            return ';'
        return str(self.expression) + ';'


class FunctionExpr(Expr):
    """
    A function
    """

    def __init__(self, declarator, parameters):
        assert isinstance(declarator, Expr)
        assert isinstance(parameters, ParenExpr)

        Expr.__init__(self, [declarator, parameters])
        self.declarator = declarator
        self.parameters = parameters

    @property
    def tokens(self):
        for t in self.parameters.tokens:
            assert isinstance(t, Token)
        return self.declarator.tokens + self.parameters.tokens

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
        tokens = [self.left_brace]
        for d in self.declarations:
            tokens += d.tokens
        for s in self.statements:
            tokens += s.tokens
        tokens.append(self.right_brace)
        return tokens

    def __str__(self):
        s = '{\n'
        if len(self.declarations) > 0:
            s += ''.join(str(d) for d in self.declarations)
        if len(self.statements) > 0:
            if len(self.declarations) > 0:
                s += '\n'
            s += '\n'.join(str(s) for s in self.statements)
            s += '\n'
        s += '}'
        return s


class InitializerListExpr(Expr, AbstractBraceExpr):
    def __init__(self, left_brace, initializers, right_brace):
        assert isinstance(initializers, CommaListExpr)
        AbstractBraceExpr.__init__(self, left_brace, right_brace)
        Expr.__init__(self, [initializers])
        self._initializers = initializers

    @property
    def initializers(self):
        return self._initializers

    @property
    def tokens(self):
        return ([self.left_brace] +
                self.initializers.tokens +
                [self.right_brace])

    def __str__(self):
        return '{' + str(self.initializers) + '}'


class FunctionDefinitionExpr(Expr):
    """
    A function definition
    """

    def __init__(self, type_expr, declarator, compound):
        """
        Warning: if the function returns a pointer, `declarator`
        is a PointerExpr.
        Moreover, type_expr is None if the return type of the
        function is unspecified.
        """

        if type_expr is not None:
            assert isinstance(type_expr, TypeExpr)
        assert isinstance(compound, CompoundExpr)

        children = []
        if type_expr is not None:
            children.append(type_expr)
        children += [declarator, compound]

        Expr.__init__(self, children)
        self.type_expr = type_expr
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

    def __str__(self):
        s = str(self.type_expr)
        if len(s) > 0:
            s += ' '
        s += str(self.declarator)
        s += '\n' + str(self.compound)
        return s


class TypeNameExpr(Expr):
    """
    Used in casts and in sizeofs
    """

    def __init__(self, type_expr, declarator):
        children = [type_expr]
        if declarator is not None:
            children.append(declarator)
        Expr.__init__(self, children)
        assert isinstance(type_expr, TypeExpr)
        self.type_expr = type_expr
        self.declarator = declarator

    def __str__(self):
        s = str(self.type_expr)
        if self.declarator is not None:
            if not isinstance(self.declarator, SubscriptExpr):
                s += ' '
            s += str(self.declarator)
        return s


class CompoundLiteralExpr(Expr):
    """
    This is a C99 feature.

    More details here:
    https://gcc.gnu.org/onlinedocs/gcc/Compound-Literals.html
    """

    def __init__(self, paren_type, compound):
        assert isinstance(paren_type, ParenExpr)
        assert isinstance(paren_type.expression, TypeNameExpr)
        assert isinstance(compound, InitializerListExpr)
        super().__init__([paren_type, compound])
        self._paren_type = paren_type
        self._compound = compound

    @property
    def paren_type(self):
        return self._paren_type

    @property
    def compound(self):
        return self._compound

    def __str__(self):
        return str(self.paren_type) + str(self.compound)


class CastExpr(Expr):
    def __init__(self, paren_type, right):
        assert isinstance(paren_type, ParenExpr)
        assert isinstance(paren_type, Expr)

        Expr.__init__(self, [paren_type, right])
        self._paren_type = paren_type
        self._right = right

    @property
    def paren_type(self):
        return self._paren_type

    @property
    def right(self):
        return self._right

    def __str__(self):
        return str(self.paren_type) + str(self.right)


class DeclarationExpr(Expr):
    def __init__(self, type_expr, declarators, semicolon):
        """
        declarators: A CommaListExpr
        """
        assert isinstance(type_expr, TypeExpr)
        assert isinstance(declarators, CommaListExpr)

        super().__init__([type_expr, declarators])
        self._type_expr = type_expr
        self._declarators = declarators
        self._semicolon = semicolon

    @property
    def type_expr(self):
        return self._type_expr

    @property
    def declarators(self):
        return self._declarators

    @property
    def semicolon(self):
        return self._semicolon

    @property
    def tokens(self):
        tokens = self.type_expr.tokens
        tokens += self.declarators.tokens
        tokens.append(self.semicolon)
        return tokens

    def __str__(self):
        s = str(self.type_expr)
        declarators = str(self.declarators)
        if len(declarators) > 0:
            s += ' ' + declarators
        return s + ';\n'


class BinaryOperationExpr(Expr):
    def __init__(self, left, operator, right):
        Expr.__init__(self, [left, right])
        self.left = left
        self.operator = operator
        self.right = right

    @property
    def tokens(self):
        return self.left.tokens + [self.operator] + self.right.tokens

    def __str__(self):
        op = self.operator.string
        if op in '. ->'.split():
            return '{}{}{}'.format(self.left, op, self.right)
        s = '{} {} {}'.format(self.left, op, self.right)
        if op != '==' and op != '!=' and '=' in op:
            return s
        return '(' + s + ')'


class TernaryOperationExpr(Expr):
    def __init__(self, left, question, mid, colon, right):
        assert question.string == '?'
        assert colon.string == ':'
        Expr.__init__(self, [left, mid, right])
        self.left = left
        self.question = question
        self.mid = mid
        self.colon = colon
        self.right = right

    @property
    def tokens(self):
        return (self.left.tokens +
                [self.question] +
                self.mid.tokens +
                [self.colon] +
                self.right.tokens)

    def __str__(self):
        return '{} ? {} : {}'.format(self.left, self.mid, self.right)


class CallExpr(Expr, AbstractParenExpr):
    def __init__(self, expression, left_paren, arguments, right_paren):
        assert isinstance(arguments, CommaListExpr)
        AbstractParenExpr.__init__(self, left_paren, right_paren)
        Expr.__init__(self, [expression, arguments])
        self._expression = expression
        self._arguments = arguments

    @property
    def expression(self):
        return self._expression

    @property
    def arguments(self):
        return self._arguments

    @property
    def tokens(self):
        return (self.expression.tokens +
                [self.left_paren] +
                self.arguments.tokens +
                [self.right_paren])

    def __str__(self):
        s = str(self.expression)
        s += '(' + str(self.arguments) + ')'
        return s


class SizeofExpr(Expr):
    def __init__(self, sizeof, expression):
        assert sizeof.string == 'sizeof'
        Expr.__init__(self, [expression])
        self.sizeof = sizeof
        self.expression = expression

    @property
    def tokens(self):
        return [self.sizeof] + self.expression.tokens

    def __str__(self):
        s = 'sizeof'
        if not isinstance(self.expression, ParenExpr):
            s += ' '
        s += str(self.expression)
        return s


class NameDesignatorExpr(Expr):
    def __init__(self, dot, identifier):
        assert isinstance(dot, Token)
        assert dot.string == '.'
        assert isinstance(identifier, LiteralExpr)
        self._dot = dot
        self._identifier = identifier
        super().__init__([])

    @property
    def dot(self):
        return self._dot

    @property
    def identifier(self):
        return self._identifier

    @property
    def tokens(self):
        return [self.dot] + self.identifier.tokens

    def __str__(self):
        return '.' + str(self.identifier)


class BracketExpr(Expr, AbstractBracketExpr):
    def __init__(self, left_bracket, expression, right_bracket):
        AbstractBracketExpr.__init__(self, left_bracket, right_bracket)
        Expr.__init__(self, [expression])
        self._expression = expression

    @property
    def expression(self):
        return self._expression

    @property
    def tokens(self):
        return ([self.left_bracket] +
                self.expression.tokens +
                [self.right_bracket])

    def __str__(self):
        return '[' + str(self.expression) + ']'


class IndexDesignatorExpr(BracketExpr):
    def __init__(self, left_bracket, expression, right_bracket):
        super().__init__(left_bracket, expression, right_bracket)


class SubscriptExpr(Expr, AbstractBracketExpr):
    def __init__(self, expression, left_bracket, index, right_bracket):
        """
        `expression` can be null in the case of a direct abstract
        declarator
        """
        AbstractBracketExpr.__init__(self, left_bracket, right_bracket)
        children = []
        if expression is not None:
            children.append(expression)
        if index is not None:
            children.append(index)
        Expr.__init__(self, children)
        self.expression = expression
        self.index = index

    @property
    def tokens(self):
        l = []
        if self.expression is not None:
            l += self.expression.tokens
        l.append(self.left_bracket)
        if self.index is not None:
            l += self.index.tokens
        l.append(self.right_bracket)
        return l

    def __str__(self):
        s = ''
        if self.expression is not None:
            s += str(self.expression)
        s += '['
        if self.index is not None:
            s += str(self.index)
        return s + ']'


class PointerExpr(Expr):
    """
    Represents a pointer declaration.

    This class cannot extend UnaryOperationExpr since `right` can be None.
    """
    def __init__(self, star, right, type_qualifiers):
        assert star.string == '*'
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
        s = '*'
        s += ' '.join(q.string for q in self.type_qualifiers)
        if len(self.type_qualifiers) > 0:
            s += ' '
        if self.right is not None:
            s += str(self.right)
        s += ''
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
        if self.postfix:
            return self.right.tokens + [self.operator]
        else:
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


class StringsExpr(LiteralExpr):
    def __init__(self, strings):
        for s in strings:
            assert isinstance(s, LiteralExpr)
            assert s.kind == 'string'
        LiteralExpr.__init__(self, strings[0])
        Expr.__init__(self, strings)

    @property
    def kind(self):
        return 'string'

    @property
    def tokens(self):
        l = []
        for c in self.children:
            l += c.tokens
        return l

    def __str__(self):
        return ' '.join(str(c) for c in self.children)

    def __repr__(self):
        return Expr.__repr__(self)


class WhileExpr(Expr):
    def __init__(self, while_token, expression, statement):
        assert isinstance(expression, ParenExpr)
        Expr.__init__(self, [expression, statement])
        self.while_token = while_token
        self.expression = expression
        self.statement = statement

    @property
    def tokens(self):
        tokens = [self.while_token]
        tokens += self.expression.tokens
        tokens += self.statement.tokens
        return tokens

    def __str__(self):
        s = 'while ' + str(self.expression) + '\n'
        s += str(self.statement)
        return s


class IfExpr(Expr):
    def __init__(self, if_token, expression, statement,
                 else_token, else_statement):
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
        tokens = [self.if_token]
        tokens += self.expression.tokens
        tokens += self.statement.tokens
        if self.else_token is not None:
            tokens.append(self.else_token)
            tokens += self.else_statement.tokens
        return tokens

    def __str__(self):
        s = 'if ' + str(self.expression) + '\n'
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


class JumpExpr(Expr):
    def __init__(self, keyword, expression):
        """
        expression: The expression after a `return` or a `goto`,
        None otherwise.
        """

        assert isinstance(keyword, Token)
        assert keyword.string in 'goto continue break return'.split()
        if keyword.string not in 'goto return':
            assert expression is None

        Expr.__init__(self, [] if expression is None else [expression])
        self.keyword = keyword
        self.expression = expression

    @property
    def tokens(self):
        t = [self.keyword]
        if self.expression is not None:
            t += self.expression.tokens
        return t

    def __str__(self):
        s = self.keyword.string
        if self.expression is not None:
            s += ' ' + str(self.expression)
        return s


class ParenExpr(Expr, AbstractParenExpr):
    def __init__(self, left_paren, expression, right_paren):
        AbstractParenExpr.__init__(self, left_paren, right_paren)
        Expr.__init__(self, [expression])
        self._expression = expression

    @property
    def expression(self):
        return self._expression

    @property
    def tokens(self):
        return [self.left_paren] + self.expression.tokens + [self.right_paren]

    def __str__(self):
        return '(' + str(self.expression) + ')'


def get_declarator_identifier(declarator):
    """
    Returns a token or None if the declarator is abstract
    """
    if isinstance(declarator, LiteralExpr):
        if declarator.kind == 'identifier':
            return declarator.token
    if isinstance(declarator, PointerExpr):
        return get_declarator_identifier(declarator.right)
    if isinstance(declarator, SubscriptExpr):
        return get_declarator_identifier(declarator.expression)
    if isinstance(declarator, FunctionExpr):
        return get_declarator_identifier(declarator.declarator)
    if isinstance(declarator, ParenExpr):
        return get_declarator_identifier(declarator.expression)
    return None


def get_declarator_name(declarator):
    """
    Returns a string or None if the declarator is abstract
    """
    token = get_declarator_identifier(declarator)
    return None if token is None else token.string


def get_makefile_content(directory_path):
    """
    Looks for a Makefile in the directory and reads it.

    Returns a string of the Makefile content on success.
    If the Makefile is not found or if an error occured,
    returns None.
    """

    makefile_path = os.path.join(directory_path, 'Makefile')
    if not os.path.exists(makefile_path):
        return None
    try:
        return open(makefile_path).read()
    except OSError:
        return None


def get_include_dirs_from_makefile(directory_path):
    """
    Returns a list of the include directores read from the Makefile.
    """
    makefile_content = get_makefile_content(directory_path)
    if makefile_content is None:
        return []
    dirs = []
    for line in makefile_content.splitlines():
        if line.count('-I') != 1:
            continue
        i = line.index('-I')
        words = line[i+2:].split()
        if words == 0:
            continue
        path = os.path.normpath(os.path.join(directory_path, words[0]))
        dirs.append(path)
    return dirs


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


def backtrack(function):
    def wrapper(*args, **kwds):
        self = args[0]
        assert isinstance(self, Parser)
        begin = self.index
        expr = function(*args, **kwds)
        if expr is None:
            self.index = begin
        return expr
    return wrapper


class Parser(TokenReader):
    """
    A parser for the C language.

    Grammar: http://www.lysator.liu.se/c/ANSI-C-grammar-y.html
    """

    def __init__(self, tokens):
        TokenReader.__init__(self, tokens)
        self._types = []
        self._included_headers = []
        self._in_typedef = False
        inc_dirs = get_include_dirs_from_makefile(self.directory_path)
        self._include_directories = inc_dirs

    def add_include_directory(self, dir_path):
        self._include_directories.append(dir_path)

    def add_include_directories(self, dirs_paths):
        for dir_path in dirs_paths:
            self.add_include_directory(dir_path)

    @property
    def include_directories(self):
        return self._include_directories[:]

    @property
    def types(self):
        return self._types[:]

    def add_type(self, type_name):
        """
        If the type is already defined, it is not added to the list.
        We can't warn or raise an error when type redefinition
        occurs since we have a no preprocessor to handle `#pragma once`
        and other directives.
        """
        if type_name not in self.types:
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

        # http://pubs.opengroup.org/onlinepubs/009696699/basedefs/sys/types.h.html
        sys_types = """
        blkcnt_t
        blksize_t
        clock_t
        clockid_t
        dev_t
        fd_set
        fsblkcnt_t
        fsfilcnt_t
        gid_t
        id_t
        ino_t
        key_t
        mode_t
        nlink_t
        off_t
        pid_t
        pthread_attr_t
        pthread_barrier_t
        pthread_barrierattr_t
        pthread_cond_t
        pthread_condattr_t
        pthread_key_t
        pthread_mutex_t
        pthread_mutexattr_t
        pthread_once_t
        pthread_rwlock_t
        pthread_rwlockattr_t
        pthread_spinlock_t
        pthread_t
        size_t
        ssize_t
        suseconds_t
        time_t
        timer_t
        trace_attr_t
        trace_event_id_t
        trace_event_set_t
        trace_id_t
        uid_t
        useconds_t
        """

        stddef = 'size_t ptrdiff_t'
        strings = {
            'assert.h':         '',
            'dirent.h':         'DIR',
            'errno.h':          '',
            'fcntl.h':          '',
            'grp.h':            '',
            'pwd.h':            '',
            'setjmp.h':         'jmp_buf',
            'signal.h':         'sighandler_t sigset_t',
            'stdarg.h':         'va_list',
            'stddef.h':         stddef,
            'stdint.h':         stdint,
            'stdio.h':          stddef + ' FILE',
            'stdlib.h':         stddef + ' div_t ldiv_t',
            'string.h':         stddef,
            'time.h':           stddef + ' clock_t time_t',
            'unistd.h':         stddef + ' ssize_t',
            'sys/stat.h':       '',
            'sys/types.h':      sys_types,
        }
        return {h: types.split() for h, types in strings.items()}

    @property
    def file_name(self):
        return self._tokens[0].begin.file_name

    @property
    def directory_path(self):
        return os.path.dirname(self.file_name)

    def _get_header_path(self, header_name, system):
        """
        system: True if the header has been included with `#include <...>`,
        false if `#include "..."` has been used.
        """

        if not system:
            header_path = os.path.join(self.directory_path, header_name)
            header_path = os.path.normpath(header_path)
            if os.path.exists(header_path):
                return header_path

        for dir_path in self.include_directories:
            if os.path.exists(dir_path):
                header_path = os.path.join(dir_path, header_name)
                header_path = os.path.normpath(header_path)
                if os.path.exists(header_path):
                    return header_path
        return None

    def _expand_local_header(self, header_path):
        if (header_path, False) in self._included_headers:
            return
        self._included_headers.append((header_path, False))

        assert os.path.exists(header_path)
        with open(header_path) as h:
            source = h.read()
        tokens = lex(source, file_name=header_path)
        parser = Parser(tokens)
        parser.add_types(self.types)
        parser.add_include_directories(self.include_directories)
        parser._included_headers += self._included_headers
        try:
            parser.parse()
        except SyntaxError as e:
            msg = e.message
            print("In file included from {}:".format(self.file_name))
            raise e
        self.add_types(parser._types)

    def _expand_system_header(self, header_name):
        headers_types = self.get_system_headers_types()
        if header_name in headers_types:

            if (header_name, True) in self._included_headers:
                return
            self._included_headers.append((header_name, True))

            self.add_types(headers_types[header_name])
            return
        print('Warning: Header not found: {!r}'.format(header_name))

    def process_include(self, directive_string):
        s = directive_string
        assert s.startswith('include')
        s = s[len('include'):].strip()
        if s.startswith('<') and s.endswith('>'):
            system = True
        elif s.startswith('"') and s.endswith('"'):
            system = False
        else:
            self.raise_syntax_error('Invalid #include directive')
        name = s[1:-1]
        header_path = self._get_header_path(name, system)
        if header_path is None:
            self._expand_system_header(name)
        else:
            self._expand_local_header(header_path)

    def has_more_impl(self, index=-1):
        if index == -1:
            index = self.index
        if index >= len(self._tokens):
            return False
        next_token = self._tokens[index]
        if next_token.kind == 'comment' or next_token.kind == 'directive':
            return self.has_more_impl(index + 1)
        return True

    def process_directive(self, directive):
        s = directive.string.strip()
        assert s[0] == '#'
        # There may be several spaces between the '#' and the
        # next word, we need to strip these
        s = s[1:].strip()
        if s.startswith('include'):
            self.process_include(s)

    @property
    def has_more(self):
        return self.has_more_impl()

    def next(self):
        token = TokenReader.next(self)
        if token.kind == 'comment':
            return self.next()
        elif token.kind == 'directive':
            self.process_directive(token)
            return self.next()
        return token

    def raise_syntax_error(self, message='Syntax error'):
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
        return None

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

    @backtrack
    def parse_identifier_token(self):
        token = self.parse_token('identifier')
        if token is None:
            return token
        if token.string in self.types and not self._in_typedef:
            return None
        return token

    def expect_identifier_token(self):
        id = self.parse_identifier_token()
        if id is None:
            self.raise_syntax_error('Expected identifier')

    def parse_identifier(self):
        token = self.parse_identifier_token()
        if token is None:
            return None
        return LiteralExpr(token)

    def parse_parameter_declaration(self):
        type_expr = self.parse_declaration_specifiers()
        if type_expr is None:
            return None
        declarator = self.parse_declarator(False)
        if declarator is None:
            declarator = self.parse_declarator(True)
        # declarator can be None
        return ParameterExpr(type_expr, declarator)

    def parse_parameter_type_list(self):
        left_paren = self.parse_sign('(')
        if left_paren is None:
            return None
        params_commas = []
        comma = None
        while self.has_more:
            if len(params_commas) > 0:
                comma = self.parse_sign(',')
                if comma is None:
                    break
                params_commas.append(comma)
            param = self.parse_parameter_declaration()
            if param is None:
                if len(params_commas) > 0:
                    param = self.parse_sign('...')
                    if param is None:
                        self.raise_syntax_error("Expected parameter after ','")
                    params_commas.append(param)
                break
            params_commas.append(param)
        right_paren = self.expect_sign(')')
        return ParenExpr(left_paren,
                         CommaListExpr(params_commas),
                         right_paren)

    @backtrack
    def parse_declarator_parens(self, abstract=False):
        left_paren = self.parse_sign('(')
        if left_paren is not None:
            if not self.has_more:
                self.raise_syntax_error("Expected ')'")
            decl = self.parse_declarator(abstract)
            if decl is not None:
                right_paren = self.expect_sign(')')
                return ParenExpr(left_paren, decl, right_paren)
        return None

    @backtrack
    def parse_declarator_brackets(self, left):
        left_bracket = self.parse_sign('[')
        if left_bracket is not None:
            if not self.has_more:
                raise_syntax_error("Expected ']'")
            constant = self.parse_constant_expression()
            if constant is None:
                right_bracket = self.parse_sign(']')
            else:
                right_bracket = self.expect_sign(']')
            if right_bracket is not None:
                return SubscriptExpr(left,
                                     left_bracket, constant, right_bracket)
        return None

    def parse_direct_abstract_declarator(self):
        left = self.parse_declarator_parens(True)
        if left is None:
            left = self.parse_declarator_brackets(None)
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

    @backtrack
    def parse_declarator(self, abstract=False):
        """
        Returns a declarator or None
        """
        star = self.parse_sign('*')
        if star is None:
            return self.parse_direct_declarator(abstract)
        type_qualifiers = self.parse_type_qualifier_list()
        right = self.parse_declarator(abstract)
        if right is None and not abstract:
            return None
        return PointerExpr(star, right, type_qualifiers)

    def parse_strings(self):
        strings = []
        while self.has_more:
            token = self.parse_token('string')
            if token is None:
                break
            strings.append(LiteralExpr(token))
        if len(strings) == 0:
            return None
        return StringsExpr(strings)

    def parse_primary_token(self):
        if not self.has_more:
            return None
        begin = self.index
        token = self.next()
        if token.kind in 'integer float character'.split():
            return token
        self.index = begin
        return self.parse_identifier_token()

    @backtrack
    def parse_paren_expression(self):
        left_paren = self.parse_sign('(')
        if left_paren is None:
            return None
        expr = self.parse_expression()
        if expr is None:
            return None
        return ParenExpr(left_paren, expr, self.expect_sign(')'))

    def parse_primary_expression(self):
        strings = self.parse_strings()
        if strings:
            return strings

        token = self.parse_primary_token()
        if token is not None:
            return LiteralExpr(token)
        return self.parse_paren_expression()

    def parse_name_designator(self):
        dot = self.parse_sign('.')
        if dot is None:
            return None
        name = self.parse_identifier()
        if name is None:
            self.raise_syntax_error()
        return NameDesignatorExpr(dot, name)

    def parse_designator(self):
        name_des = self.parse_name_designator()
        if name_des is not None:
            return name_des
        left_bracket = self.parse_sign('[')
        if left_bracket is None:
            return None
        index = self.parse_constant_expression()
        if index is None:
            self.raise_syntax_error()
        right_bracket = self.parse_sign(']')
        return IndexDesignatorExpr(left_bracket, index, right_bracket)

    def parse_designation(self):
        designator = self.parse_designator()
        if designator is None:
            return None
        eq = self.expect_sign('=')
        right = self.parse_initializer()
        if right is None:
            self.raise_syntax_error()
        return BinaryOperationExpr(designator, eq, right)

    def parse_initializer_list(self):
        left_brace = self.parse_sign('{')
        if left_brace is None:
            return None
        l = []
        while True:
            if len(l):
                comma = self.parse_sign(',')
                if comma is None:
                    break
                l.append(comma)
            init = self.parse_designation()
            if init is None:
                init = self.parse_initializer()
                if init is None:
                    self.raise_syntax_error('Expected initializer')
            l.append(init)
        right_brace = self.expect_sign('}')
        list_expr = CommaListExpr(l)
        return InitializerListExpr(left_brace, list_expr, right_brace)

    def parse_initializer(self):
        init_list = self.parse_initializer_list()
        if init_list is not None:
            return init_list
        expr = self.parse_assignment_expression()
        if expr is not None:
            return expr
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
                self.raise_syntax_error('Initializer expected')
            return BinaryOperationExpr(declarator, eq, initializer)
        return declarator

    def parse_init_declarator_list(self):
        """
        Returns a CommaListExpr
        """
        declarators = []
        comma = None
        while self.has_more:
            declarator = self.parse_init_declarator()
            if declarator is None:
                if comma is None:
                    break
                else:
                    self.raise_syntax_error("Expected declarator after ','",
                                            comma.begin)
            declarators.append(declarator)
            comma = self.parse_sign(',')
            if comma is None:
                break
            declarators.append(comma)
        return CommaListExpr(declarators)

    def parse_storage_class_specifier(self):
        """
        Returns a TypeSpecifierExpr or None
        """
        kw_strings = 'typedef extern static auto register'.split()
        token = self.parse_keyword(kw_strings)
        if token is None:
            return None
        return TypeSpecifierExpr(token)

    def get_type_specifiers_strings(self):
        return 'void char short int long float double signed unsigned'.split()

    def parse_struct_or_union_specifier(self):
        kw = self.parse_keyword('struct union'.split())
        if kw is None:
            return None
        identifier = self.parse_identifier_token()
        compound = self.parse_compound_statement()
        if compound is not None:
            if len(compound.statements) > 0:
                self.raise_syntax_error('Expected type name')
        if identifier is None and compound is None:
            self.raise_syntax_error("Expected identifier or "
                                    "'{}' after {!r}".format('{', kw.string))
        return StructExpr(kw, identifier, compound)
        # TODO

    def parse_enumerator(self):
        return self.parse_identifier()

    def parse_enumerator_list(self):
        left_brace = self.parse_sign('{')
        if left_brace is None:
            return None
        l = []
        while True:
            enumerator = self.parse_enumerator()
            if enumerator is None:
                break
            l.append(enumerator)
            comma = self.parse_sign(',')
            if comma is None:
                break
            l.append(comma)
        right_brace = self.expect_sign('}')
        list_expr = CommaListExpr(l, allow_trailing=True)
        return InitializerListExpr(left_brace, list_expr, right_brace)

    def parse_enum_specifier(self):
        kw = self.parse_keyword(['enum'])
        if kw is None:
            return None
        identifier = self.parse_identifier_token()
        body = self.parse_enumerator_list()
        if identifier is None and body is None:
            self.raise_syntax_error("Expected identifier or "
                                    "'{}' after {!r}".format('{', kw.string))
        return EnumExpr(kw, identifier, body)

    @backtrack
    def parse_type_specifier(self, allowed_type_specs='bc'):
        """
        Returns an AbstractTypeSpecifierExpr or None
        """

        if 'b' in allowed_type_specs:
            kw = self.parse_keyword(self.get_type_specifiers_strings())
            if kw is not None:
                return TypeSpecifierExpr(kw)

        if 'c' not in allowed_type_specs:
            return None

        struct = self.parse_struct_or_union_specifier()
        if struct is not None:
            return struct

        enum = self.parse_enum_specifier()
        if enum is not None:
            return enum

        token = self.parse_token('identifier')
        if token is not None and token.string in self.types:
            return TypeSpecifierExpr(token)
        return None

    def parse_type_qualifier(self):
        """
        Returns a TypeSpecifierExpr or None
        """
        kw_strings = ('const volatile '.split())
        kw = self.parse_keyword(kw_strings)
        return None if kw is None else TypeSpecifierExpr(kw)

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

    def _parse_type_specifiers_list(self,
                                    allow_storage_class_specs,
                                    allowed_type_specs='bc'):
        """
        Returns a list of AbstractTypeSpecifierExpr
        """
        specs = []
        if allow_storage_class_specs:
            spec = self.parse_storage_class_specifier()
            if spec is not None:
                specs.append(spec)

        spec = self.parse_type_specifier(allowed_type_specs)
        if spec is not None:
            specs.append(spec)
            is_keyword = (isinstance(spec, TypeSpecifierExpr) and
                          spec.token.kind == 'keyword')
            if is_keyword:
                allowed_type_specs = 'b'
            else:
                allowed_type_specs = ''

        spec = self.parse_type_qualifier()
        if spec is not None:
            specs.append(spec)

        if len(specs) == 0:
            return specs
        others = self._parse_type_specifiers_list(allow_storage_class_specs,
                                                  allowed_type_specs)
        return specs + others

    def _parse_type_specifiers(self, allow_storage_class_specifiers):
        """
        Returns a TypeExpr or None
        """
        specs = self._parse_type_specifiers_list(
            allow_storage_class_specifiers)
        for spec in specs:
            assert isinstance(spec, AbstractTypeSpecifierExpr)
        if len(specs) == 0:
            return None
        return TypeExpr(specs)

    def parse_declaration_specifiers(self):
        """
        Returns a TypeExpr or None
        """
        return self._parse_type_specifiers(True)

    def parse_specifier_qualifier_list(self):
        """
        Returns a TypeExpr or None
        """
        return self._parse_type_specifiers(False)

    def parse_type_name(self):
        specifiers = self.parse_specifier_qualifier_list()
        if specifiers is None:
            return None
        return TypeNameExpr(specifiers, self.parse_declarator(abstract=True))

    def filter_type_specifiers(self, tokens):
        return [t for t in tokens if
                t.string in self.get_type_specifiers_strings()]

    def _add_type_from_declarator(self, decl):
        name = get_declarator_name(decl)
        if name is None:
            msg = ('Cannot retrieve the name of the declarator '
                   '{!r} (repr: {!r})'.format(str(decl), decl))
            raise Exception(msg)
        self.add_type(name)

    @backtrack
    def parse_declaration(self):
        type_expr = self.parse_declaration_specifiers()
        if type_expr is None:
            return None
        if type_expr.is_typedef:
            self._in_typedef = True
        declarators = self.parse_init_declarator_list()
        self._in_typedef = False
        semicolon = self.parse_sign(';')
        if semicolon is None:
            return None
        if type_expr.is_typedef:
            if len(declarators) == 0:
                self.raise_syntax_error("Expected type name after 'typedef'")
            for decl in declarators.children:
                self._add_type_from_declarator(decl)
        return DeclarationExpr(type_expr, declarators, semicolon)

    def parse_declaration_list(self):
        declarations = []
        while True:
            decl = self.parse_declaration()
            if decl is None:
                break
            declarations.append(decl)
        return declarations

    def parse_argument_expression_list(self):
        args_commas = []
        while True:
            if len(args_commas) > 0:
                comma = self.parse_sign(',')
                if comma is None:
                    break
                args_commas.append(comma)
            argument = self.parse_assignment_expression()
            if argument is None:
                argument = self.parse_type_name()
                if argument is None:
                    if len(args_commas) == 0:
                        break
                    self.raise_syntax_error('Expected argument')
            args_commas.append(argument)
        return CommaListExpr(args_commas)

    @backtrack
    def parse_compound_literal(self):
        parens = self.parse_parenthesed_type_name()
        if parens is None:
            return None
        compound = self.parse_initializer_list()
        if compound is None:
            return None
        return CompoundLiteralExpr(parens, compound)

    def parse_postfix_expression(self):
        left = self.parse_primary_expression()
        if left is None:
            left = self.parse_compound_literal()
            if left is None:
                return None
        while True:
            operator = self.parse_sign('[ ( ++ -- . ->'.split())
            if operator is None:
                break
            if operator.string == '(':
                args_commas = self.parse_argument_expression_list()
                right_paren = self.expect_sign(')')
                left = CallExpr(left, operator, args_commas, right_paren)
            elif operator.string == '[':
                expr = self.parse_expression()
                right_bracket = self.expect_sign(']')
                left = SubscriptExpr(left, operator, expr, right_bracket)
            elif operator.string in '++ --'.split():
                left = UnaryOperatorerationExpr(operator, left, postfix=True)
            elif operator.string in '. ->'.split():
                identifier = self.parse_identifier()
                if identifier is None:
                    self.raise_syntax_error('Expected an identifier')
                left = BinaryOperationExpr(left, operator, identifier)
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
            return SizeofExpr(sizeof, expr)
        left_paren = self.expect_sign('(')
        type_name = self.parse_type_name()
        if type_name is None:
            self.raise_syntax_error('Expected type name')
        right_paren = self.expect_sign(')')
        return SizeofExpr(sizeof,
                          ParenExpr(left_paren, type_name, right_paren))

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

    @backtrack
    def parse_parenthesed_type_name(self):
        left_paren = self.parse_sign('(')
        if left_paren is None:
            return None
        type_name = self.parse_type_name()
        if type_name is None:
            return None
        right_paren = self.expect_sign(')')
        if right_paren is None:
            return None
        return ParenExpr(left_paren, type_name, right_paren)

    def parse_cast_expression(self):
        expr = self.parse_unary_expression()
        if expr is not None:
            return expr
        parens = self.parse_parenthesed_type_name()
        if parens is None:
            return None
        expr = self.parse_cast_expression()
        if expr is None:
            return self.raise_syntax_error('Expected expression')
        return CastExpr(parens, expr)

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

    def parse_and_expression(self):
        return self.parse_binary_operation('&',
                                           self.parse_equality_expression)

    def parse_exclusive_or_expression(self):
        return self.parse_binary_operation('^',
                                           self.parse_and_expression)

    def parse_inclusive_or_expression(self):
        return self.parse_binary_operation('|',
                                           self.parse_exclusive_or_expression)

    def parse_logical_and_expression(self):
        return self.parse_binary_operation('&&',
                                           self.parse_inclusive_or_expression)

    def parse_logical_or_expression(self):
        return self.parse_binary_operation('||',
                                           self.parse_logical_and_expression)

    def parse_conditional_expression(self):
        left = self.parse_logical_or_expression()
        quest = self.parse_sign('?')
        if quest is None:
            return left
        mid = self.parse_expression()
        if mid is None:
            self.raise_syntax_error('Expected expression')
        colon = self.expect_sign(':')
        right = self.parse_conditional_expression()
        return TernaryOperationExpr(left, quest, mid, colon, right)

    def parse_constant_expression(self):
        return self.parse_conditional_expression()

    def parse_assignment_operator(self):
        """
        Returns an assignment operator or None
        """
        ops = '= *= /= %= += -= <<= >>= &= ^= |='.split()
        return self.parse_sign(ops)

    @backtrack
    def parse_assignment_expression_2(self):
        unary = self.parse_unary_expression()
        if unary is None:
            return None
        op = self.parse_assignment_operator()
        if op is None:
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
            self.raise_syntax_error("The 'switch' statement is not "
                                    "implemented")

        if_token = self.parse_keyword(['if'])
        if if_token is None:
            return None
        left_paren = self.expect_sign('(')
        expr = self.parse_expression()
        if expr is None:
            self.raise_syntax_error('Expected expression')
        right_paren = self.expect_sign(')')
        statement = self.parse_statement()
        if statement is None:
            self.raise_syntax_error('Expected statement')

        else_statement = None
        else_token = self.parse_keyword(['else'])
        if else_token is not None:
            else_statement = self.parse_statement()
            if else_statement is None:
                self.raise_syntax_error("Expected statement after 'else'")
        return IfExpr(if_token,
                      ParenExpr(left_paren, expr, right_paren), statement,
                      else_token, else_statement)

    def parse_iteration_statement(self):
        token = self.parse_keyword('do for'.split())
        if token is not None:
            self.raise_syntax_error("'do' and 'for' statements are not "
                                    "implemented")
        while_token = self.parse_keyword('while'.split())
        if while_token is None:
            return None
        left_paren = self.expect_sign('(')
        expr = self.parse_expression()
        if expr is None:
            self.raise_syntax_error('Expected expression')
        right_paren = self.expect_sign(')')
        statement = self.parse_statement()
        if statement is None:
            self.raise_syntax_error('Expected statement')
        return WhileExpr(while_token,
                         ParenExpr(left_paren, expr, right_paren),
                         statement)

    def parse_return_statement(self):
        return_token = self.parse_keyword(['return'])
        if return_token is None:
            return None
        expr = None
        semicolon = self.parse_sign(';')
        if semicolon is None:
            expr = self.parse_expression()
            if expr is None:
                self.raise_syntax_error('Expected expression')
            semicolon = self.expect_sign(';')
        return StatementExpr(JumpExpr(return_token, expr), semicolon)

    def parse_jump_statement(self):
        r = self.parse_return_statement()
        if r is not None:
            return r
        token = self.parse_keyword(['break', 'continue'])
        if token is None:
            return None
        semicolon = self.expect_sign(';')
        return StatementExpr(JumpExpr(token, None), semicolon)

    def parse_statement(self):
        stmt = self.parse_compound_statement()
        if stmt is not None:
            return stmt
        stmt = self.parse_expression_statement()
        if stmt is not None:
            return stmt
        stmt = self.parse_selection_statement()
        if stmt is not None:
            return stmt
        stmt = self.parse_iteration_statement()
        if stmt is not None:
            return stmt
        stmt = self.parse_jump_statement()
        if stmt is not None:
            return stmt
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
        type_expr = self.parse_declaration_specifiers()
        declarator = self.parse_declarator()
        if declarator is None:
            if type_expr is None:
                return None
            self.raise_syntax_error('Expected declarator')
        compound = self.parse_compound_statement()
        if compound is None:
            self.raise_syntax_error('Expected compound statement')
        return FunctionDefinitionExpr(type_expr, declarator, compound)

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
                s = 'Unexpected {!r}'.format(self.next().string)
                self.raise_syntax_error(s)
            if decl is not None:
                declarations.append(decl)
        expr = TranslationUnitExpr(declarations)
        while self.index < len(self._tokens):
            token = self._tokens[self.index]
            if token.kind == 'directive':
                self.process_directive(token)
            self.index += 1
        return expr

    def parse(self):
        return self.parse_translation_unit()


def argument_to_tokens(v):
    if isinstance(v, str):
        v = lex(v)
    if not isinstance(v, list):
        raise ArgumentError('Expected a list of tokens')
    return v


def parse(v, include_directories=None):
    """
    v: a string or a list of tokens
    """

    if include_directories is None:
        include_directories = []

    tokens = argument_to_tokens(v)
    parser = Parser(tokens)
    parser.add_include_directories(include_directories)
    tree = parser.parse()
    assert isinstance(tree, Expr)
    tokens = [t for t in tokens if t.kind not in 'comment directive'.split()]
    if tokens != tree.tokens:
        if isinstance(v, str):
            print(v)
        print(tree)
        print()
        print('\n'.join(repr(t) for t in tokens))
        print()
        print('\n'.join(repr(t) for t in tree.tokens))
    assert tokens == tree.tokens
    return tree


def parse_statement(v):
    """
    v: a string or a list of tokens
    """
    parser = Parser(argument_to_tokens(v))
    return parser.parse_statement()


def parse_expr(v):
    """
    v: a string or a list of tokens
    Can return None
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
        with self.assertRaises(SyntaxError):
            parse_expr('a.""')
        with self.assertRaises(SyntaxError):
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

        with self.assertRaises(SyntaxError):
            parse('struct s {a + a;}')

    def test_compound_literal(self):
        self.checkExpr('(int *){23, 45, 67}')
        self.checkExpr('(int[]){23, 45, 67}')
        self.checkExpr('(struct s *[12]){23, 45, 67}')

    def test_initializer_list(self):
        self.checkDecl('int a[] = {2, 3, 4};')
        pass

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

        with self.assertRaises(SyntaxError):
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
        e = parse_expr('1 + 1')
        plus = e.select('binary_operation')
        assert len(plus) == 1
        assert isinstance(list(plus)[0], BinaryOperationExpr)

        one = e.select('literal')
        assert len(one) == 2
        for c in one:
            assert isinstance(c, LiteralExpr)

        one = e.select('binary_operation literal')
        assert len(one) == 2
        for c in one:
            assert isinstance(c, LiteralExpr)

        with self.assertRaises(ValueError):
            parse_expr('123').select('')

        with self.assertRaises(ValueError):
            parse_expr('123').select('eiaueiuaeiua')


class StyleIssue(AbstractIssue):
    def __init__(self, message, position, level='error'):
        assert level in 'warn error'.split()
        super().__init__(message, position)
        self._level = level

    @property
    def level(self):
        return self._level


class StyleChecker:
    def __init__(self, issue_handler):
        self._issue_handler = issue_handler

    def issue(self, issue):
        assert isinstance(issue, StyleIssue)
        self._issue_handler(issue)

    def warn(self, message, position):
        self.issue(StyleIssue(message, position, level='warn'))

    def error(self, message, position):
        self.issue(StyleIssue(message, position))

    def check(self, tokens, expr):
        raise Exception('Not implemented')


class SourceChecker(StyleChecker):
    """
    Like a StyleChecker, but accepts a string containing the source
    code of the file
    """
    def __init__(self, issue_handler):
        super().__init__(issue_handler)

    def check(self, source, tokens, expr):
        raise Exception('Not implemented')


class LineChecker(SourceChecker):
    def __init__(self, issue_handler):
        super().__init__(issue_handler)

    def check_line(self, begin, line, end):
        raise Exception()

    def check(self, source, tokens, expr):
        lines = source.splitlines(True)
        index = 0
        file_name = tokens[0].begin.file_name
        for line_index, line_and_endl in enumerate(lines):
            line = line_and_endl.rstrip('\n\r')
            begin = Position(file_name,
                             index,
                             line_index + 1)
            end_col_index = max(0, len(line) - 1)
            end = Position(file_name,
                           index + end_col_index,
                           line_index + 1,
                           end_col_index)
            self.check_line(begin, line, end)
            index += len(line_and_endl)


class LineLengthChecker(LineChecker):
    def __init__(self, issue_handler, options):
        super().__init__(issue_handler)

    def check_line(self, begin, line, end):
        if len(line) > 80:
            self.error("Line too long (more than 80 characters)", end)


class TrailingWhitespaceChecker(LineChecker):
    def __init__(self, issue_handler, options):
        super().__init__(issue_handler)

    def check_line(self, begin, line, end):
        if line.rstrip() != line:
            self.warn("Trailing whitespaces at the end of the line", end)


class HeaderCommentChecker(StyleChecker):
    def __init__(self, issue_handler, options):
        self.username_check_enabled = options.header_username
        super().__init__(issue_handler)

    def check_username(self, login, position):
        if not self.username_check_enabled:
            return
        if not re.match(r'\w+_\w', login):
            self.warn("Not a valid EPITECH username (was {!r})".format(login),
                      position)

    def check_line(self, i, line, position):
        if i == 3:
            login = line.split()[-1]
            self.check_username(login, position)
            if not line.startswith("** Made by"):
                self.error("Invalid 'Made by' line", position)
        if i == 6:
            login = line.split()[-1]
            self.check_username(login, position)
            if not line.startswith("** Started on"):
                self.error("Invalid 'Started on' line", position)
        if i == 7:
            login = line.split()[-1]
            self.check_username(login, position)
            if not line.startswith("** Last update"):
                self.error("Invalid 'Last update' line", position)

    def check_header(self, token):
        lines = token.string.splitlines()
        if len(lines) != 9:
            self.error('The header must be 9 lines long', token.begin)
        for i, line in enumerate(lines):
            self.check_line(i, line, token.begin)

    def check(self, tokens, expr):
        for token in tokens:
            if token.kind == 'comment' and token.begin.line == 1:
                self.check_header(token)
                return
        self.error("No header comment", tokens[0].begin)


class CommentChecker(StyleChecker):
    def __init__(self, issue_handler, options):
        super().__init__(issue_handler)

    def get_all_tokens_in_expr(self, tokens, expr):
        first = expr.first_token
        last = expr.last_token
        all_tokens = tokens[tokens.index(first):tokens.index(last) + 1]
        assert first == all_tokens[0]
        assert last == all_tokens[-1]
        return all_tokens

    def check_comment(self, comment):
        if comment.begin.column != 1:
            self.error("A comment must begin at the first column",
                       comment.begin)

        lines = comment.string.splitlines()
        if len(lines) < 3:
            self.error("A comment must be at least 3 lines long",
                       comment.begin)
            return

        if lines[0] != '/*':
            self.error("A comment must start with '/*'", comment.begin)
        if lines[-1] != '*/':
            self.error("A comment must end with '*/'", comment.begin)
        for line in lines[1:-1]:
            if line[:3].strip() != '**':
                self.error("The comment lines should start with '**'",
                           comment.begin)

    def check(self, tokens, expr):
        for t in tokens:
            if t.kind == 'comment':
                self.check_comment(t)

        funcs = expr.select('function_definition')
        for func in funcs:
            func_tokens = self.get_all_tokens_in_expr(tokens, func)
            for token in func_tokens:
                if token.kind == 'comment':
                    self.error('Comment inside a function', token.begin)


class MarginChecker(StyleChecker):
    def __init__(self, issue_handler):
        super().__init__(issue_handler)

    def check_margin(self, left_token, margin, right_token):
        left_end = left_token.end
        right_begin = right_token.begin
        if left_end.line != right_begin.line:
            return
        if left_end.column + margin + 1 != right_begin.column:
            m = 'Expected {} space(s) between {!r} and {!r}'.format(
                margin, left_token.string, right_token.string)
            self.error(m, left_end)


class BinaryOpSpaceChecker(MarginChecker):
    def __init__(self, issue_handler, options):
        super().__init__(issue_handler)

    def check(self, tokens, expr):
        bin_ops = expr.select('binary_operation')
        for operation in bin_ops:
            left_token = operation.left.last_token
            right_token = operation.right.first_token
            op = operation.operator
            margin = 0 if op.string in '. ->'.split() else 1
            self.check_margin(left_token, margin, op)
            self.check_margin(op, margin, right_token)


class UnaryOpSpaceChecker(MarginChecker):
    def __init__(self, issue_handler, options):
        super().__init__(issue_handler)

    def check(self, tokens, expr):
        unary_ops = expr.select('unary_operation')
        for operation in unary_ops:
            if operation.postfix:
                left_token = operation.right.last_token
                op = operation.operator
                self.check_margin(left_token, 0, op)
            else:
                right_token = operation.right.first_token
                op = operation.operator
                self.check_margin(op, 0, right_token)


class ReturnChecker(StyleChecker):
    def __init__(self, issue_handler, options):
        super().__init__(issue_handler)

    def check_return(self, return_expr):
        if not isinstance(return_expr.expression, ParenExpr):
            self.error("No paretheses after 'return'",
                       return_expr.keyword.end)

    def check(self, tokens, expr):
        returns = expr.select('jump')
        for return_expr in returns:
            if return_expr.keyword.string == 'return':
                if return_expr.expression is not None:
                    self.check_return(return_expr)


class FunctionLengthChecker(StyleChecker):
    def __init__(self, issue_handler, options):
        super().__init__(issue_handler)

    def check(self, tokens, expr):
        funcs = expr.select('function_definition')
        for func in funcs:
            begin = func.compound.left_brace
            begin_line = begin.begin.line
            end_line = func.compound.right_brace.begin.line
            line_count = end_line - begin_line - 1
            if line_count > 25:
                self.error('Too long function (more than 25 lines)',
                           func.compound.left_brace.begin)
                return
            elif line_count > 24:
                self.warn('Long function ({} lines)'.format(line_count),
                          func.compound.left_brace.begin)


class FunctionCountChecker(StyleChecker):
    def __init__(self, issue_handler, options):
        super().__init__(issue_handler)

    def check(self, tokens, expr):
        funcs = expr.select('function_definition')
        if len(funcs) > 5:
            self.error('Too many functions (more than 5)',
                       expr.first_token.begin)


class DirectiveIndentationChecker(StyleChecker):
    def __init__(self, issue_handler, options):
        super().__init__(issue_handler)

    def get_indent_level(self, string, position):
        level = 0
        for c in string:
            if c == '\t':
                self.warn("Tabulation after '#'", position)
            elif c == ' ':
                level += 1
            else:
                break
        return level

    def bad_indent_level(self, current, expected, position):
        self.error("Bad indent level (expected {} space(s), got {})".format(
            expected, current), position)

    def check(self, tokens, expr):
        level = 0
        for token in tokens:
            if token.kind == 'directive':
                string = token.string.strip()[1:]
                name = string.strip()
                if name.startswith('endif'):
                    level -= 1
                directive_level = self.get_indent_level(string, token.begin)
                if name.startswith('else') or name.startswith('elif'):
                    directive_level += 1
                if level != directive_level:
                    self.bad_indent_level(directive_level, level, token.begin)
                if name.startswith('if'):
                    level += 1


class DeclarationChecker(StyleChecker):
    def __init__(self, issue_handler, options):
        super().__init__(issue_handler)

    def check_same_line(self, left, right):
        assert isinstance(left, Token)
        assert isinstance(right, Token)
        if left.end.line != right.end.line:
            msg = "{!r} is not on the same line than {!r}".format(
                left.string, right.string)
            self.error(msg, left.begin)

    def check_struct(self, struct):
        if struct.identifier is not None:
            self.check_same_line(struct.struct, struct.identifier)

    def check_declaration(self, decl):
        if len(decl.declarators.children) > 0:
            self.check_same_line(decl.declarators.last_token,
                                 decl.semicolon)

    def check_function_def(self, func_def):
        if func_def.type_expr is not None:
            self.check_same_line(func_def.type_expr.last_token,
                                 func_def.declarator.first_token)

    def check_new_line_constistency(self, expr):
        children = expr.children
        prev = None
        for child in children:
            if prev is not None and len(prev.tokens) and len(child.tokens):
                self.check_same_line(prev.last_token, child.first_token)
            prev = child

    def check(self, tokens, expr):
        structs = expr.select('struct')
        for struct in structs:
            self.check_struct(struct)
        decls = expr.select('declaration')
        for decl in decls:
            self.check_declaration(decl)
            self.check_new_line_constistency(decl)
        types = expr.select('type')
        for t in types:
            self.check_new_line_constistency(t)
        funcs = expr.select('function_definition')
        for func in funcs:
            self.check_function_def(func)
        decls = expr.select('function')
        for decl in decls:
            self.check_new_line_constistency(decl)


def get_argument_parser():
    descr = 'Check your C programs against the "EPITECH norm".'
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('source_files',
                        nargs='*',
                        help='source files or directories to check')
    parser.add_argument('--test', action='store_true',
                        help='run the old tests')
    parser.add_argument('--header-username', action='store_true',
                        help="check the username in header comments")
    parser.add_argument('-I', action='append',
                        help="add a directory to the header search path")
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


def print_escape_code(code, string, end):
    print('\x1b[' + str(code) + 'm' + string + '\x1b[0m', end=end)


def get_color_code(name):
    colors = 'black red green yellow blue magenta cyan white'.split()
    if name not in colors:
        raise Exception()
    return colors.index(name)


def print_fg_color(color_name, string, end='\n'):
    print_escape_code(get_color_code(color_name) + 90, string, end)


def print_issue(issue):
    color = 'red' if issue.level == 'error' else 'yellow'
    if os.isatty(sys.stdout.fileno()):
        print_fg_color(color, issue.level, end=' ')
    else:
        print(issue.level, end=' ')
    print(issue)


class Program:
    """
    The main program
    """

    def __init__(self, args):
        self.include_dirs = args.I
        if self.include_dirs is None:
            self.include_dirs = []
        checkers_classes = [
            BinaryOpSpaceChecker,
            CommentChecker,
            DeclarationChecker,
            DirectiveIndentationChecker,
            FunctionLengthChecker,
            FunctionCountChecker,
            HeaderCommentChecker,
            LineLengthChecker,
            ReturnChecker,
            TrailingWhitespaceChecker,
            UnaryOpSpaceChecker,
        ]
        self.colors = os.isatty(sys.stdout.fileno())
        self.checkers = [c(print_issue, args) for c in checkers_classes]

    def check_file_or_dir(self, path, include_dirs=None):
        if include_dirs is None:
            include_dirs = self.include_dirs
            if os.path.isdir(path):
                include_dirs += get_include_dirs_from_makefile(path)

        if os.path.isdir(path):
            self._check_dir(path, include_dirs)
        elif os.path.isfile(path):
            self._check_file(path, include_dirs)
        else:
            raise Exception('{!r} is not a file nor a directory'.format(path))

    def _check_dir(self, path, include_dirs):
        include_dirs += get_include_dirs_from_makefile(path)
        for file_name in os.listdir(path):
            if file_name.startswith('.'):
                continue
            file_path = os.path.join(path, file_name)
            if os.path.isfile(file_path):
                if not file_name.endswith('.c'):
                    continue
            self.check_file_or_dir(file_path, include_dirs)

    def _print_fg_color(self, color_name, string):
        if self.colors:
            print_fg_color(color_name, string)
        else:
            print(string)

    def _check_file(self, path, include_dirs):
        self._print_fg_color('black', path)
        with open(path) as source_file:
            source = source_file.read()
            tokens = lex(source, file_name=path)
            root_expr = parse(tokens, include_dirs)
            for checker in self.checkers:
                if isinstance(checker, SourceChecker):
                    checker.check(source, tokens, root_expr)
                else:
                    checker.check(tokens, root_expr)


def main():
    argument_parser = get_argument_parser()
    args = argument_parser.parse_args()
    if args.test:
        test()
        return

    program = Program(args)
    for source_file in args.source_files:
        program.check_file_or_dir(source_file)


if __name__ == '__main__':
    main()
