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
        system_include_pattern = r'^include\s+<[\w\./]+>$'
        local_include_pattern = r'^include\s+"[\w\./]+"$'
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
        if isinstance(self, cls):
            return [self]
        l = []
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

    def __init__(self, comma_separated_children):
        """
        comma_separated_children is a list of expressions
        separated by commas tokens.
        No trailing comma.
        """

        if len(comma_separated_children) > 0:
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


def makeAbstractBracketExpr(class_name, signs, name):
    """
    Returns a new class
    """
    def __init__(self, left_bracket, right_bracket):
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


AbstractParenExpr = makeAbstractBracketExpr('AbstractParenExpr',
                                            '()', 'paren')

AbstractBracketExpr = makeAbstractBracketExpr('AbstractBracketExpr',
                                              '[]', 'bracket')

AbstractBraceExpr = makeAbstractBracketExpr('AbstractBraceExpr',
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

    def __init__(self, struct_token, identifier, compound):
        """
        struct_token: The keyword `struct` or `union`
        identifier: The name of the struct or None
        compound: The "body" of the struct or None
        """
        assert isinstance(struct_token, Token)
        assert (struct_token.string == 'struct' or
                struct_token.string == 'union')

        if identifier is not None:
            assert isinstance(identifier, Token)
            assert identifier.kind == 'identifier'

        if compound is not None:
            assert isinstance(compound, CompoundExpr)
            assert len(compound.statements) == 0

        Expr.__init__(self, [] if compound is None else [compound])

        self._struct_token = struct_token
        self._identifier = identifier
        self._compound = compound

    @property
    def kind(self):
        """
        Returns 'struct' or 'union'
        """
        return self.struct_token.string

    @property
    def struct_token(self):
        return self._struct_token

    @property
    def identifier(self):
        return self._identifier

    @property
    def compound(self):
        return self._compound

    @property
    def tokens(self):
        t = [self.struct_token]
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
        return (self.declarator.tokens + self.parameters.tokens)

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


class FunctionDefinitionExpr(Expr):
    """
    A function definition
    """

    def __init__(self, type_expr, declarator, compound):
        """
        Warning: if the function returns a pointer, `declarator`
        is a PointerExpr.
        """

        assert isinstance(type_expr, TypeExpr)
        assert isinstance(compound, CompoundExpr)

        Expr.__init__(self, [type_expr, declarator, compound])
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
            s += ' ' + str(self.declarator)
        return s


class CastExpr(Expr):
    def __init__(self, paren_type, right):
        assert isinstance(paren_type, ParenExpr)

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
        return str(self.paren_type) + ' ' + str(self.right)


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


class SubscriptExpr(Expr, AbstractBracketExpr):
    def __init__(self, expression, left_bracket, index, right_bracket):
        AbstractBracketExpr.__init__(self, left_bracket, right_bracket)
        children = [expression]
        if index is not None:
            children.append(index)
        Expr.__init__(self, children)
        self.expression = expression
        self.index = index

    @property
    def tokens(self):
        return (self.expression.tokens +
                [self.left_bracket] +
                (self.index.tokens if self.index is not None else []) +
                [self.right_bracket])

    def __str__(self):
        s = str(self.expression) + '['
        if self.index is not None:
            s += str(self.index)
        return s + ']'


class PointerExpr(Expr):
    """
    Represents a pointer declaration.

    This class cannot extend UnaryOperationExpr since `right` can be None.
    """
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
        self.expression = expression

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
            'dirent.h':         'DIR',
            'setjmp.h':         'jmp_buf',
            'stddef.h':         stddef,
            'stdint.h':         stdint,
            'stdio.h':          stddef + ' FILE',
            'stdlib.h':         stddef + ' div_t ldiv_t',
            'string.h':         stddef,
            'time.h':           stddef + ' clock_t time_t',
            'unistd.h':         stddef + ' ssize_t',
            'sys/types.h':      sys_types,
        }
        return {h: types.split() for h, types in strings.items()}

    def _expand_local_header(self, header_name):
        import os
        file_name = self._tokens[0].begin.file_name
        dir_name = os.path.dirname(file_name)
        header_path = os.path.join(dir_name, header_name)
        print('local header path:', header_path)
        if not os.path.exists(header_path):
            return self.expand_header(header_name, True)
        with open(header_path) as h:
            source = h.read()
        tokens = lex(source, file_name=header_path)
        parser = Parser(tokens)
        parser._types = self._types[:]
        try:
            parser.parse()
        except SyntaxError as e:
            msg = e.message
            print(msg)
        self.add_types(parser._types)

    def expand_header(self, header_name, system):
        if (header_name, system) in self._included_headers:
            return
        self._included_headers.append((header_name, system))
        headers_types = self.get_system_headers_types()
        if header_name in headers_types:
            self.add_types(headers_types[header_name])
        else:
            self._expand_local_header(header_name)

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
                    self.raise_syntax_error("Expected parameter after ','")
                break
            params_commas.append(param)
        right_paren = self.expect_sign(')')
        return ParenExpr(left_paren,
                         CommaListExpr(params_commas),
                         right_paren)

    def parse_declarator_parens(self, abstract=False):
        begin = self.index
        left_paren = self.parse_sign('(')
        if left_paren is not None:
            if not self.has_more:
                self.raise_syntax_error("Expected ')'")
            decl = self.parse_declarator(abstract)
            if decl is not None:
                right_paren = self.expect_sign(')')
                return ParenExpr(left_paren, decl, right_paren)
            self.index = begin
        return None

    def parse_declarator_brackets(self, left):
        begin = self.index
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
                                    "'{' after {!r}".format(kw.string))
        return StructExpr(kw, identifier, compound)
        # TODO

    def parse_type_specifier(self):
        """
        Returns an AbstractTypeSpecifierExpr or None
        """
        kw = self.parse_keyword(self.get_type_specifiers_strings())
        if kw is not None:
            return TypeSpecifierExpr(kw)

        struct = self.parse_struct_or_union_specifier()
        if struct is not None:
            return struct

        begin = self.index
        token = self.parse_token('identifier')
        if token is not None and token.string in self.types:
            return TypeSpecifierExpr(token)
        self.index = begin

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

    def _parse_type_specifiers_list(self, allow_storage_class_specs):
        """
        Returns a list of AbstractTypeSpecifierExpr
        """
        specs = []
        if allow_storage_class_specs:
            spec = self.parse_storage_class_specifier()
            if spec is not None:
                specs.append(spec)

        spec = self.parse_type_specifier()
        if spec is not None:
            specs.append(spec)

        spec = self.parse_type_qualifier()
        if spec is not None:
            specs.append(spec)

        if len(specs) == 0:
            return specs
        return specs + self._parse_type_specifiers_list(
            allow_storage_class_specs)

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
        if len(specifiers) == 0:
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
        self.add_type(get_declarator_name(decl))

    def parse_declaration(self):
        begin = self.index
        type_expr = self.parse_declaration_specifiers()
        if type_expr is None:
            return None
        declarators = self.parse_init_declarator_list()
        semicolon = self.parse_sign(';')
        if semicolon is None:
            self.index = begin
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
                if len(args_commas) == 0:
                    break
                self.raise_syntax_erorr('Expected argument')
            args_commas.append(argument)
        return CommaListExpr(args_commas)

    def parse_postfix_expression(self):
        left = self.parse_primary_expression()
        if left is None:
            return None
        while True:
            op = self.parse_sign('[ ( ++ -- . ->'.split())
            if op is None:
                break
            if op.string == '(':
                args_commas = self.parse_argument_expression_list()
                right_paren = self.expect_sign(')')
                left = CallExpr(left, op, args_commas, right_paren)
            elif op.string == '[':
                expr = self.parse_expression()
                right_bracket = self.expect_sign(']')
                left = SubscriptExpr(left, op, expr, right_bracket)
            elif op.string in '++ --'.split():
                left = UnaryOperationExpr(op, left, postfix=True)
            elif op.string in '. ->'.split():
                identifier = self.parse_identifier()
                if identifier is None:
                    self.raise_syntax_error('Expected an identifier')
                left = BinaryOperationExpr(left, op, identifier)
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
        parens = ParenExpr(left_paren, type_name, right_paren)
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
    tokens = argument_to_tokens(v)
    parser = Parser(tokens)
    tree = parser.parse()
    tokens = [t for t in tokens if t.kind not in 'comment directive'.split()]
    if len(tokens) != len(tree.tokens):
        if isinstance(v, str):
            print(v)
        print(tree)
        print()
        print('\n'.join(repr(t) for t in tokens))
        print()
        print('\n'.join(repr(t) for t in tree.tokens))
    assert len(tokens) == len(tree.tokens)
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

    def test_sizeof(self):
        self.checkExpr('sizeof(int)')
        self.checkExpr('sizeof(123)')
        self.checkExpr('sizeof 123')

    def test_cast(self):
        self.checkExpr('(int) 54.9')
        self.checkExpr('(char **) 17')

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

    def test_enum(self):
        # TODO
        # self.checkDecl('enum s;')
        pass

    def test_typedef(self):
        with self.assertRaises(SyntaxError):
            parse('a b;')

        self.checkDecl('typedef int (*a);\n\n'
                       'typedef a b;\n\n'
                       'b c;')
        self.checkDecl('typedef int a[32];')
        self.checkDecl('typedef int (*a)();')
        self.checkDecl('typedef int a;')

    def test_typedef_struct(self):
        self.checkDecl('typedef struct s_dir\n'
                       '{\n'
                       'int n;\n'
                       'int m;\n'
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
        assert type(plus[0]) is BinaryOperationExpr

        one = e.select('literal')
        assert len(one) == 2
        for c in one:
            assert type(c) is LiteralExpr

        one = e.select('binary_operation literal')
        assert len(one) == 2
        for c in one:
            assert type(c) is LiteralExpr

        with self.assertRaises(ValueError):
            parse_expr('123').select('')

        with self.assertRaises(ValueError):
            parse_expr('123').select('eiaueiuaeiua')


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
    tokens = lex(source, file_name=source_file.name)
    # print('\n'.join(repr(t) for t in tokens))
    print('parsing...')
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
