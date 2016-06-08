#!/usr/bin/env python3

# pylint: disable=too-many-lines, redefined-variable-type, abstract-method

"""
A C brace style checker designed for the "EPITECH norm".
"""

import argparse
import os
import re
import sys


COPYRIGHT = """
Copyright © 2016 Antoine Motet <antoine.motet@epitech.eu>

This work is free. You can redistribute it and/or modify it under the
terms of the Do What The Fuck You Want To Public License, Version 2,
as published by Sam Hocevar. See http://www.wtfpl.net/ for more details.
"""


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
        """
        Return the file path.
        """
        return self._file_name

    @property
    def index(self):
        """
        Return the index relative to the begin of the file.
        """
        return self._index

    @property
    def line(self):
        """
        Returns the 1-based line number.
        """
        return self._line

    @property
    def column(self):
        """
        Return the 1-based column number.
        """
        return self._column

    def __add__(self, other):
        """
        Add an integer to the index of this position.

        Don't mutate this position and returns an integer.
        """
        return self.index + other

    def __sub__(self, other):
        """
        Subtract an integer or the index of another position from this
        position.

        Don't mutate this position and returns an integer.
        """
        if isinstance(other, Position):
            other = other.index
        return self.index - other

    def __str__(self):
        """
        Return an user-friendly string describing this position.
        """
        return '{}:{}:{}'.format(
            self.file_name,
            self.line,
            self.column)


TOKEN_KINDS = [
    'identifier',
    'integer', 'float', 'string', 'character',
    'sign',
    'keyword',
    'comment',
    'directive',
]


class Token:
    """
    Represents a token
    """

    def __init__(self, kind, string, begin, end):
        """
        kind: A string describing the kind of the token. It shoud be an
        item of TOKEN_KINDS.
        string: The string of the token. This is a part of the source,
        its length should be equal to `end - begin`.
        begin: A Position representing the begin of the token
        end: A Position representing the end of the token
        """

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
        """
        Return a string describing the kind of the token.

        TOKEN_KINDS contains the list of the different kinds.
        """
        return self._kind

    @property
    def string(self):
        """
        Return the string of the token.
        """
        return self._string

    @property
    def begin(self):
        """
        Return a Position describing the begin of the token.
        """
        return self._begin

    @property
    def end(self):
        """
        Return a Position describing the end of the token.
        """
        return self._end

    def __str__(self):
        """
        Return the string of the token.
        """
        return self.string

    def __repr__(self):
        """
        Return a string describing the token for debugging purposes.
        """
        return '<Token kind={}, string={!r}, begin={}, end={}>'.format(
            self.kind, self.string, self.begin, self.end,
        )


class StyleIssue:
    """
    Represents a style issue reported by the checker.
    """

    def __init__(self, message, position, level='error'):
        """
        message: A string describing the issue
        position: The Position of the issue
        """
        assert level in 'note warn error syntax-error'.split()
        assert isinstance(message, str)
        assert isinstance(position, Position)
        self._level = level
        self._message = message
        self._position = position

    @property
    def message(self):
        """
        Return a string describing the issue

        Don't include the position of the issue.
        """
        return self._message

    @property
    def position(self):
        """
        Return the position of the issue
        """
        return self._position

    @property
    def level(self):
        return self._level

    def __str__(self):
        """
        Return an user-friendly string describing the issue and
        its position
        """
        return str(self.position) + ': ' + self.level + ': ' + self.message


class NSyntaxError(Exception, StyleIssue):
    """
    Represents a syntax error.
    """

    def __init__(self, message, position):
        Exception.__init__(self, message)
        StyleIssue.__init__(self, message, position, 'syntax-error')

    def __str__(self):
        return StyleIssue.__str__(self)


def raise_expected_string(expected_string, position):
    """
    Raise a syntax error when `expected_string` is expected in
    a source file, but not present.
    """
    raise NSyntaxError("Expected '{}'".format(expected_string), position)


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
noreturn
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
    """
    Return a big regexp string representing the whole lexer grammar.

    Concat the strings returned by get_lexer_spec().
    """
    def get_token_regex(pair):
        return '(?P<{}>{})'.format(*pair)

    lexer_spec = get_lexer_spec()
    return '|'.join(get_token_regex(pair) for pair in lexer_spec)


def check_directive(string, begin):
    """
    Check if a preprocessor directive is valid.

    string: A string of the directive.
    begin: The position of the first character of the directive.
    """
    string = string.strip()[1:].strip()
    if string.startswith('include'):
        system_include_pattern = r'^include\s+<[\w\./]+>$'
        local_include_pattern = r'^include\s+"[\w\./]+"$'
        if (re.match(system_include_pattern, string) is None and
                re.match(local_include_pattern, string) is None):
            msg = "Invalid #include directive (was {!r})".format('#' + string)
            raise NSyntaxError(msg, begin)


def lex_token(source_string, file_name):
    """
    A generator to tokenize the given string.

    Yields tokens.

    file_name: The source file name, used to raise precise errors.
    """
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
            raise NSyntaxError("{!r} unexpected".format(string), begin)
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
    """
    Tokenize a string.

    Returns a list of tokens.
    """
    tokens = []
    for token in lex_token(string, file_name):
        tokens.append(token)
    return tokens


class Expr:
    """
    Represents a node of the syntax tree.

    An Expr is immutable.
    """

    def __init__(self, children):
        self._children = children[:]
        for child in children:
            assert isinstance(child, Expr)

    @staticmethod
    def _split_camel_case(string):
        """
        Split the words in a camel-case-formatted string.

        Returns a list of words in lowercase.

        >>> Expr._split_camel_case('FooBar')
        ['foo', 'bar']
        """
        def find_uppercase_letter(string):
            """
            Return the index of the first uppercase letter in a string,
            or -1 if not found.
            """
            for i, char in enumerate(string):
                if char.isupper():
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
        """
        Return a short name from an expression class name.

        >>> Expr._get_class_short_name('FooBarExpr')
        'foo_bar'
        """
        name = name[:-len('Expr')]
        words = Expr._split_camel_case(name)
        return '_'.join(words)

    @staticmethod
    def _get_expr_classes():
        """
        Return a dict of the classes in this module whose name ends with
        'Expr'.
        """
        import inspect
        classes = inspect.getmembers(sys.modules[__name__], inspect.isclass)
        classes_dict = {}
        for name, cls in classes:
            if name.endswith('Expr') and name != 'Expr':
                short_name = Expr._get_class_short_name(name)
                classes_dict[short_name] = cls
        classes_dict['expr'] = Expr
        return classes_dict

    def select_classes(self, cls):
        selected = []
        if isinstance(self, cls):
            selected.append(self)
        for child in self.children:
            selected += child.select_classes(cls)
        return frozenset(selected)

    @staticmethod
    def select_all(expressions, selectors_string):
        selected = []
        for expr in expressions:
            selected += expr.select(selectors_string)
        return frozenset(selected)

    def select(self, selectors_string):
        """
        Select child nodes from a selector string, a bit like CSS
        selection.

        Return a set.
        """
        selectors = selectors_string.split()
        if len(selectors) == 0:
            raise ValueError('No selector in the given string')
        selector = selectors[0]
        expr_classes = Expr._get_expr_classes()
        selected = frozenset()
        if selector in expr_classes:
            selected = self.select_classes(expr_classes[selector])
        else:
            raise ValueError('Invalid selector: {!r}'.format(selector))
        if len(selectors) > 1:
            return Expr.select_all(selected, ' '.join(selectors[1:]))
        return selected

    @property
    def children(self):
        """
        Return the children expressions of this expression.
        """
        return self._children[:]

    @property
    def tokens(self):
        """
        Return a list of the token of this expression, including the
        tokens of the children.
        """
        tokens = []
        for child in self.children:
            for token in child.tokens:
                assert isinstance(token, Token)
            tokens += child.tokens
        return tokens

    @property
    def first_token(self):
        """
        Return the first token of this expression.
        """
        return self.tokens[0]

    @property
    def last_token(self):
        """
        Return the last token of this expression.
        """
        return self.tokens[-1]

    def __len__(self):
        """
        Return the length of the chidren expressions.
        """
        return len(self.children)

    def __str__(self):
        raise Exception('Not implemented')

    def __repr__(self):
        class_name = self.__class__.__name__
        return '<{} children={}>'.format(class_name, self.children)


class CommaListExpr(Expr):
    """
    Represents a list of expressions separated by commas.
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
    Return a new class representing a bracket pair (parentheses,
    braces or whatever).
    """
    def __init__(self, left_bracket, right_bracket):
        # pylint fails to analyse properly this function.
        # pylint: disable=protected-access

        assert isinstance(left_bracket, Token)
        assert left_bracket.kind == 'sign'
        assert left_bracket.string == signs[0]

        assert isinstance(right_bracket, Token)
        assert right_bracket.kind == 'sign'
        assert right_bracket.string == signs[1]

        self._left_bracket = left_bracket
        self._right_bracket = right_bracket

    def right_bracket(self):
        """
        Return the right bracket token.
        """
        # pylint: disable=protected-access
        return self._right_bracket

    def left_bracket(self):
        """
        Return the left bracket token.
        """
        # pylint: disable=protected-access
        return self._left_bracket

    return type(class_name, (object,), {
        '__init__': __init__,
        'left_' + name: property(left_bracket),
        'right_' + name: property(right_bracket),
    })


# pylint: disable=invalid-name

AbstractParenExpr = make_abstract_bracket_expr('AbstractParenExpr',
                                               '()', 'paren')

AbstractBracketExpr = make_abstract_bracket_expr('AbstractBracketExpr',
                                                 '[]', 'bracket')

AbstractBraceExpr = make_abstract_bracket_expr('AbstractBraceExpr',
                                               '{}', 'brace')
# pylint: enable=invalid-name


class AbstractTypeSpecifierExpr(Expr):
    """
    Represents a type specifier.
    """

    def __init__(self, children):
        Expr.__init__(self, children)


class TypeSpecifierExpr(AbstractTypeSpecifierExpr):
    """
    Represents a single token type specifier.
    """

    def __init__(self, specifier_token):
        assert (specifier_token.kind == 'identifier' or
                specifier_token.kind == 'keyword')
        super().__init__([])
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
        Return the string 'struct' or 'union'.
        """
        return self.struct.string

    @property
    def struct(self):
        """
        Return the 'struct' or 'union' token.
        """
        return self._struct

    @property
    def identifier(self):
        """
        Return the name of the struct or None.

        A struct name is not a type name. For example, this struct
        is named 'a', not 'b' :

            typedef struct a {
                int c;
            } b;
        """
        return self._identifier

    @property
    def compound(self):
        """
        Return the "body" of the struct, braces included.
        """
        return self._compound

    @property
    def tokens(self):
        tokens = [self.struct]
        if self.identifier is not None:
            tokens.append(self.identifier)
        if self.compound is not None:
            tokens += self.compound.tokens
        return tokens

    def __str__(self):
        string = self.kind
        if self.identifier is not None:
            string += ' ' + self.identifier.string
        if self.compound is not None:
            string += '\n' + str(self.compound)
        return string


class EnumExpr(AbstractTypeSpecifierExpr):
    """
    Represents an enum.
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
        """
        Return the 'enum' token.
        """
        return self._enum

    @property
    def identifier(self):
        """
        Return the token representing the name of the enum or None.
        """
        return self._identifier

    @property
    def body(self):
        """
        Return the "body" of the enum, braces included.
        """
        return self._body

    @property
    def tokens(self):
        tokens = [self.enum]
        if self.identifier is not None:
            tokens.append(self.identifier)
        if self.body is not None:
            tokens += self.body.tokens
        return tokens

    def __str__(self):
        string = 'enum'
        if self.identifier is not None:
            string += ' ' + self.identifier.string
        if self.body is not None:
            string += '\n' + str(self.body)
        return string


class TypeExpr(Expr):
    """
    Represents a type.
    """

    def __init__(self, children):
        assert len(children) > 0
        Expr.__init__(self, children)

    def has_type_specifier(self, type_specifier):
        for child in self.children:
            if isinstance(child, TypeSpecifierExpr):
                if child.token.string == type_specifier:
                    return True
        return False

    @property
    def is_typedef(self):
        return self.has_type_specifier('typedef')

    @property
    def is_extern(self):
        return self.has_type_specifier('extern')

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
        token = []
        if self.expression is not None:
            token += self.expression.tokens
        return token + [self.semicolon]

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
        for token in self.parameters.tokens:
            assert isinstance(token, Token)
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
        for decl in self.declarations:
            tokens += decl.tokens
        for statement in self.statements:
            tokens += statement.tokens
        tokens.append(self.right_brace)
        return tokens

    def __str__(self):
        string = '{\n'
        if len(self.declarations) > 0:
            string += ''.join(str(d) for d in self.declarations)
        if len(self.statements) > 0:
            if len(self.declarations) > 0:
                string += '\n'
            string += '\n'.join(str(s) for s in self.statements)
            string += '\n'
        string += '}'
        return string


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
        string = str(self.type_expr)
        if len(string) > 0:
            string += ' '
        string += str(self.declarator)
        string += '\n' + str(self.compound)
        return string


class TypeNameExpr(Expr):
    """
    A type name is a type with an optional declarator.

    Used in casts and in sizeofs.
    """

    def __init__(self, type_expr, declarator=None):
        """
        type_expr: A TypeExpr.
        declarator: A declarator or None.
        """
        children = [type_expr]
        if declarator is not None:
            children.append(declarator)
        Expr.__init__(self, children)
        assert isinstance(type_expr, TypeExpr)
        self.type_expr = type_expr
        self.declarator = declarator

    def __str__(self):
        string = str(self.type_expr)
        if self.declarator is not None:
            if not isinstance(self.declarator, SubscriptExpr):
                string += ' '
            string += str(self.declarator)
        return string


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
        string = str(self.type_expr)
        declarators = str(self.declarators)
        if len(declarators) > 0:
            string += ' ' + declarators
        return string + ';\n'


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
        operator = self.operator.string
        if operator in '. ->'.split():
            return '{}{}{}'.format(self.left, operator, self.right)
        string = '{} {} {}'.format(self.left, operator, self.right)
        if operator != '==' and operator != '!=' and '=' in operator:
            return string
        return '(' + string + ')'


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
        string = str(self.expression)
        string += '(' + str(self.arguments) + ')'
        return string


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
        string = 'sizeof'
        if not isinstance(self.expression, ParenExpr):
            string += ' '
        string += str(self.expression)
        return string


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
        tokens = []
        if self.expression is not None:
            tokens += self.expression.tokens
        tokens.append(self.left_bracket)
        if self.index is not None:
            tokens += self.index.tokens
        tokens.append(self.right_bracket)
        return tokens

    def __str__(self):
        string = ''
        if self.expression is not None:
            string += str(self.expression)
        string += '['
        if self.index is not None:
            string += str(self.index)
        return string + ']'


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
        # pylint: disable=no-member
        string = '*'
        string += ' '.join(qual.string for qual in self.type_qualifiers)
        if len(self.type_qualifiers) > 0:
            string += ' '
        if self.right is not None:
            string += str(self.right)
        string += ''
        return string


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


class AbstractLiteralExpr(Expr):
    """
    A literal expression.
    """

    def __init__(self, children):
        super().__init__(children)


class LiteralExpr(Expr):
    """
    A literal expression with one token.
    """

    def __init__(self, token):
        super().__init__([])
        literals = 'identifier integer float string character'.split()
        assert token.kind in literals
        self.token = token

    @property
    def kind(self):
        """
        Return the kind of the underlying token.
        """
        return self.token.kind

    @property
    def tokens(self):
        return [self.token]

    @property
    def string(self):
        """
        Return the string of the underlying token.
        """
        return self.token.string

    def __str__(self):
        return self.token.string

    def __repr__(self):
        return '<LiteralExpr kind={} token={!r}>'.format(
            self.kind, self.string)


class StringsExpr(AbstractLiteralExpr):
    """
    A concatenation of multiple strings.
    """

    def __init__(self, strings):
        for string in strings:
            assert isinstance(string, LiteralExpr)
            assert string.kind == 'string'
        super().__init__(strings)

    @property
    def tokens(self):
        tokens = []
        for child in self.children:
            tokens += child.tokens
        return tokens

    def __str__(self):
        return ' '.join(str(child) for child in self.children)

    def __repr__(self):
        return Expr.__repr__(self)


class WhileExpr(Expr):
    """
    A `while` statement.
    """

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
        string = 'while ' + str(self.expression) + '\n'
        string += str(self.statement)
        return string


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
        string = 'if ' + str(self.expression) + '\n'
        string += str(self.statement)
        if self.else_token is not None:
            string += '\nelse\n'
            string += str(self.else_statement)
        return string


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
        token = [self.keyword]
        if self.expression is not None:
            token += self.expression.tokens
        return token

    def __str__(self):
        string = self.keyword.string
        if self.expression is not None:
            string += ' ' + str(self.expression)
        return string


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
        self._tokens = tokens[:]
        self._index = 0

    @property
    def tokens(self):
        return self._tokens[:]

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


def _get_system_header_types():
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

    # shared by curses.h and ncurses.h
    curses = 'MEVENT WINDOW'

    stddef = 'size_t ptrdiff_t'
    strings = {
        'assert.h':     '',
        'curses.h':     curses,
        'dirent.h':     'DIR',
        'errno.h':      '',
        'fcntl.h':      '',
        'glob.h':       'glob_t',
        'grp.h':        '',
        'ncurses.h':    curses,
        'pwd.h':        '',
        'setjmp.h':     'jmp_buf',
        'signal.h':     'sighandler_t sigset_t',
        'stdarg.h':     'va_list',
        'stdbool.h':    'bool',
        'stddef.h':     stddef,
        'stdint.h':     stdint,
        'stdio.h':      stddef + ' FILE',
        'stdlib.h':     stddef + ' div_t ldiv_t',
        'string.h':     stddef,
        'termios.h':    'pid_t',
        'time.h':       stddef + ' clock_t time_t',
        'unistd.h':     stddef + ' ssize_t',
        'sys/stat.h':   '',
        'sys/types.h':  sys_types,
    }
    return {h: types.split() for h, types in strings.items()}


class IncludedFile:
    """
    Represents a file included with an #include directive.
    """

    def __init__(self, system, path):
        assert isinstance(system, bool)
        assert isinstance(path, str)
        self._system = system
        self._path = path
        if system:
            self._types = _get_system_header_types()[path]
        else:
            self._types = None

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
                self.__dict__ == other.__dict__)

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def system(self):
        return self._system

    @property
    def types(self):
        return self._types

    @types.setter
    def types(self, types):
        assert isinstance(types, list)
        if self.types is not None:
            raise Exception('The types of this file are already set')
        if self._system:
            raise Exception('The types of a system header file are read-only')
        self._types = types

    @property
    def path(self):
        return self._path

    def __repr__(self):
        return '<IncludedFile system={!r} path={!r} types={!r}>'.format(
            self.system, self.path, self.types)


class IncludedFileCache:
    def __init__(self):
        self._files = []

    def file(self, path):
        for inc_file in self._files:
            if inc_file.path == path:
                return inc_file
        return None

    def add_file(self, included_file):
        assert not included_file.system
        if included_file in self._files:
            raise Exception('File already cached')
        self._files.append(included_file)


class YanDirective:
    """
    Represents a Yan comment directive.
    """

    def __init__(self, token_index):
        """
        token_index: The index of the directive in the preprocessed token
        list.
        """
        self._token_index = token_index

    @property
    def token_index(self):
        return self._token_index


class YanTypedefDirective(YanDirective):
    def __init__(self, token_index, type_names):
        assert isinstance(type_names, list)
        super().__init__(token_index)
        self._type_names = type_names[:]

    @property
    def type_names(self):
        return self._type_names[:]


class FileInclusion(YanDirective):
    """
    Represents an inclusion with an #include directive.
    """

    def __init__(self, token_index, inc_file):
        """
        token_index: The index of the removed inclusion in the
        preprocessed token list.
        inc_file: An IncludedFile
        """
        super().__init__(token_index)
        assert isinstance(inc_file, IncludedFile)
        self._file = inc_file

    @property
    def file(self):
        return self._file

    def __repr__(self):
        return '<FileInclusion file={!r} token_index={!r}>'.format(
            self.file, self.token_index)


class PreprocessorResult:
    def __init__(self, tokens, directives, issues):
        self._tokens = tokens[:]
        self._directives = directives[:]
        self._issues = issues[:]

    @property
    def tokens(self):
        return self._tokens[:]

    @property
    def directives(self):
        return self._directives[:]

    @property
    def issues(self):
        return self._issues[:]

    def append(self, o):
        if isinstance(o, Token):
            self._tokens.append(o)
        elif isinstance(o, YanDirective):
            self._directives.append(o)
        else:
            raise ValueError()

    def __add__(self, other):
        return PreprocessorResult(self.tokens + other.tokens,
                                  self.directives + other.directives,
                                  self.issues + other.issues)


def get_included_file_path(dir_path,
                           included_file_name, system, include_dirs):
    """
    Searches a file name in the given directories.

    dir_path: The path of the directory of the source file.
    included_file_name: The included header name.
    system: True if the header has been included with `#include <...>`,
    false if `#include "..."` has been used.

    Returns the path of the file to include relative to `dir_path` if this
    file is found.
    Returns None otherwise.
    """

    if not system:
        header_path = os.path.join(dir_path, included_file_name)
        header_path = os.path.normpath(header_path)
        if os.path.exists(header_path):
            return header_path

    for inc_dir in include_dirs:
        if not os.path.exists(inc_dir):
            continue
        header_path = os.path.join(inc_dir, included_file_name)
        header_path = os.path.normpath(header_path)
        if os.path.exists(header_path):
            return header_path
    return None


class Preprocessor(TokenReader):
    """
    Warning: This is not a real C preprocessor.
    """

    def __init__(self, tokens, include_dirs=None):
        super().__init__(tokens)
        if include_dirs is None:
            include_dirs = []
        assert isinstance(include_dirs, list)
        self._include_dirs = include_dirs[:]
        self._issues = []

    @property
    def issues(self):
        return self._issues[:]

    def _warn(self, message, position=None):
        if position is None:
            position = self.position
        self._issues.append(StyleIssue(message, position, level='warn'))

    def _note(self, message, position=None):
        if position is None:
            position = self.position
        self._issues.append(StyleIssue(message, position, level='note'))

    @staticmethod
    def _get_name_and_system(directive_string, position):
        """
        Parse a directive string.

        Return a tuple (name, system) where 'name' is the included file
        name specified between the quotes, and 'system' is True if
        '#include <...>' has been used, False if '#include "..."' has
        been used.

        directive_string: A directive string without the '#'
        position: The position of that directive, used to raise a syntax
        error if the given string is invalid.
        """
        assert directive_string.startswith('include')
        quoted_name = directive_string[len('include'):].strip()
        name = quoted_name[1:-1]
        if quoted_name.startswith('<') and quoted_name.endswith('>'):
            return name, True
        elif quoted_name.startswith('"') and quoted_name.endswith('"'):
            return name, False
        else:
            raise NSyntaxError('Invalid #include directive', position)

    def _preprocess_system_include(self, name, position):
        """
        Returns an IncludedFile on success.
        Returns None if there is no system header with the given name.
        """
        system_headers = _get_system_header_types()
        if name not in system_headers:
            self._note('Header not found: {!r}'.format(name), position)
            return None
        return IncludedFile(True, name)

    @staticmethod
    def _preprocess_local_include(path):
        """
        Returns an IncludedFile.
        """
        assert os.path.exists(path)
        return IncludedFile(False, path)

    @staticmethod
    def _remove_directive_leading_hash(directive):
        assert directive.kind == 'directive'
        string = directive.string.strip()
        assert string[0] == '#'
        # There may be several spaces between the '#' and the
        # next word, we need to strip these
        return string[1:].strip()

    def _preprocess_include(self, directive):
        """
        Returns an IncludedFile or None
        """
        string = self._remove_directive_leading_hash(directive)
        name, system = self._get_name_and_system(string,
                                                 directive.begin)
        file_name = directive.begin.file_name
        dir_path = os.path.dirname(file_name)
        included_file_path = get_included_file_path(dir_path,
                                                    name, system,
                                                    self._include_dirs)
        if included_file_path is None:
            return self._preprocess_system_include(name, directive.begin)
        else:
            return self._preprocess_local_include(included_file_path)

    def _token_is_if(self, token):
        if token.kind != 'directive':
            return False
        string = self._remove_directive_leading_hash(token)
        return string.startswith('if')

    def _token_is_endif(self, token):
        if token.kind != 'directive':
            return False
        string = self._remove_directive_leading_hash(token)
        return string.startswith('endif')

    def _skip_until_endif(self):
        level = 1
        while level > 0:
            if not self.has_more:
                self._warn("Expected '#endif'")
                return
            token = self.next()
            if self._token_is_if(token):
                level += 1
            elif self._token_is_endif(token):
                level -= 1

    def _preprocess_directive(self, token_index, directive):
        """
        Return a FileInclusion or None.
        """
        string = self._remove_directive_leading_hash(directive)
        if string.startswith('include'):
            inc_file = self._preprocess_include(directive)
            if inc_file is None:
                return None
            return FileInclusion(token_index, inc_file)
        if string.startswith('ifdef'):
            if '__cplusplus' in string:
                self._skip_until_endif()
                return None
            return None

    @staticmethod
    def _get_yan_directive_comment(comment_string):
        lines = comment_string.splitlines()
        for line in lines:
            line = line.lstrip('*').strip()
            if line.startswith('yan '):
                return line[3:].strip()
        return None

    def _skip_until_parser_on(self):
        while True:
            if not self.has_more:
                return
            token = self.next()
            if token.kind != 'comment':
                continue
            directive = self._get_yan_directive_comment(token.string)
            if directive is None:
                continue
            if directive == 'parser on':
                return

    def _preprocess_comment(self, token_index, comment):
        directive = Preprocessor._get_yan_directive_comment(comment.string)
        if directive is None:
            return self._preprocess_token(token_index)
        self._note('Yan comment directive', comment.begin)
        if directive == 'parser on':
            raise NSyntaxError('The parser is already enabled', comment.begin)
        if directive == 'parser off':
            self._skip_until_parser_on()
        if directive.startswith('typedef'):
            type_names = directive.split()[1:]
            return YanTypedefDirective(token_index, type_names)
        return self._preprocess_token(token_index)

    def _preprocess_token(self, token_index):
        """
        Return a YanDirective, a Token or None.
        """
        if not self.has_more:
            return None
        token = self.next()
        if token.kind == 'comment':
            return self._preprocess_comment(token_index, token)
        if token.kind == 'directive':
            token = self._preprocess_directive(token_index, token)
            if token is None:
                return self._preprocess_token(token_index)
            return token
        return token

    def preprocess(self):
        """
        Preprocess the tokens.

        Return a PreprocessorResult.
        """
        result = PreprocessorResult([], [], [])
        while True:
            i = len(result.tokens)
            token = self._preprocess_token(i)
            if token is None:
                break
            result.append(token)
        result = result + PreprocessorResult([], [], self.issues)
        return result


def preprocess(tokens, include_dirs=None):
    """
    Warning: This is not a real C preprocessor.

    This function acts a bit like a preprocessor since it removes
    comments and expands includes.
    """
    cpp = Preprocessor(tokens, include_dirs)
    return cpp.preprocess()


def backtrack(function):
    """
    This function is used as a decorator in the parser.

    If the decorated function returns None, backtracking is performed:
    the parser current token is reset to the token when the decorated
    function was called.
    """
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

    The methods of this class reflect mostly the following grammar:
    http://www.lysator.liu.se/c/ANSI-C-grammar-y.html
    """
    # pylint: disable=invalid-name

    def __init__(self,
                 preprocessor_result,
                 file_name='<unknown file>',
                 include_dirs=None,
                 included_file_cache=None):
        """
        Create a new parser
        """
        assert isinstance(file_name, str)
        assert isinstance(preprocessor_result, PreprocessorResult)
        if included_file_cache is None:
            included_file_cache = IncludedFileCache()
        assert isinstance(included_file_cache, IncludedFileCache)
        if include_dirs is None:
            include_dirs = []
        assert isinstance(include_dirs, list)
        TokenReader.__init__(self, preprocessor_result.tokens)
        self._directives = preprocessor_result.directives[:]
        self._file_name = file_name
        self._include_dirs = include_dirs[:]
        self._types = []
        self._included_files = []
        self._included_file_cache = included_file_cache
        self._in_typedef = False
        self._issues = preprocessor_result.issues

    @property
    def issues(self):
        return self._issues[:]

    @property
    def file_name(self):
        """
        Return the file name or '<unknown file>'.
        """
        return self._file_name

    def _is_already_included(self, included_file):
        assert isinstance(included_file, IncludedFile)
        return included_file in self._included_files

    @property
    def types(self):
        """
        Return the list of the type names defined in this file (or in
        a previously included file).
        """
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

    def _expand_local_include(self, included_file):
        path = included_file.path
        assert os.path.exists(path)

        cached_file = self._included_file_cache.file(path)
        if cached_file is not None:
            included_file = cached_file

        assert not included_file.system

        if included_file.types is not None:
            self.add_types(included_file.types)
            return

        with open(path) as h:
            source = h.read()
        source_tokens = lex(source, file_name=path)
        pp_result = preprocess(source_tokens, self._include_dirs)
        parser = Parser(pp_result,
                        path,
                        self._include_dirs,
                        self._included_file_cache)
        parser.add_types(self.types)
        parser._included_files += self._included_files
        try:
            parser.parse()
        except NSyntaxError as e:
            # print("In file included from {}:".format(self.file_name))
            raise e

        included_file.types = parser.types
        self._included_file_cache.add_file(included_file)
        self.add_types(parser.types)

    def _expand_system_include(self, included_file):
        assert included_file.system
        self.add_types(included_file.types)

    def _expand_included_file(self, included_file):
        assert isinstance(included_file, IncludedFile)

        if self._is_already_included(included_file):
            return
        self._included_files.append(included_file)

        if included_file.system:
            self._expand_system_include(included_file)
        else:
            self._expand_local_include(included_file)

    def _expand_directives(self):
        while len(self._directives) > 0:
            directive = self._directives[0]
            if not isinstance(directive, YanDirective):
                continue
            if directive.token_index != self.index:
                break
            if isinstance(directive, FileInclusion):
                self._expand_included_file(directive.file)
            elif isinstance(directive, YanTypedefDirective):
                self.add_types(directive.type_names)
            else:
                raise Exception()
            self._directives.pop(0)

    def next(self):
        self._expand_directives()
        return TokenReader.next(self)

    def _raise_syntax_error(self, message='Syntax error'):
        """
        Raises a syntax error at the current position.
        """
        raise NSyntaxError(message, self.position)

    def _parse_token(self, kind, string_list=None):
        if isinstance(string_list, str):
            string_list = [string_list]
        if not self.has_more:
            return None
        begin = self.index
        token = self.next()
        if token.kind == kind:
            if (string_list is not None) and token.string in string_list:
                return token
            elif string_list is None:
                return token
        self.index = begin
        return None

    def _parse_keyword(self, keyword_list=None):
        for kw in keyword_list:
            assert kw in KEYWORDS
        return self._parse_token('keyword', keyword_list)

    def _parse_sign(self, sign_list=None):
        if sign_list is None:
            sign_list = []
        for sign in sign_list:
            assert sign in SIGNS
        return self._parse_token('sign', sign_list)

    def _expect_sign(self, sign_list):
        sign = self._parse_sign(sign_list)
        if sign is None:
            self._raise_syntax_error('Expected {!r}'.format(sign_list))
        return sign

    @backtrack
    def _parse_identifier_token(self):
        token = self._parse_token('identifier')
        if token is None:
            return token
        if token.string in self.types and not self._in_typedef:
            return None
        return token

    def _parse_identifier(self):
        token = self._parse_identifier_token()
        if token is None:
            return None
        return LiteralExpr(token)

    def _parse_parameter_declaration(self):
        type_expr = self._parse_declaration_specifiers()
        if type_expr is None:
            return None
        declarator = self._parse_declarator(False)
        if declarator is None:
            declarator = self._parse_declarator(True)
        # declarator can be None
        return ParameterExpr(type_expr, declarator)

    def _parse_parameter_type_list(self):
        left_paren = self._parse_sign('(')
        if left_paren is None:
            return None
        params_commas = []
        comma = None
        while self.has_more:
            if len(params_commas) > 0:
                comma = self._parse_sign(',')
                if comma is None:
                    break
                params_commas.append(comma)
            param = self._parse_parameter_declaration()
            if param is None:
                if len(params_commas) > 0:
                    param = self._parse_sign('...')
                    if param is None:
                        self._raise_syntax_error()
                    params_commas.append(param)
                break
            params_commas.append(param)
        right_paren = self._expect_sign(')')
        return ParenExpr(left_paren,
                         CommaListExpr(params_commas),
                         right_paren)

    @backtrack
    def _parse_declarator_parens(self, abstract=False):
        left_paren = self._parse_sign('(')
        if left_paren is not None:
            if not self.has_more:
                self._raise_syntax_error("Expected ')'")
            decl = self._parse_declarator(abstract)
            if decl is not None:
                right_paren = self._expect_sign(')')
                return ParenExpr(left_paren, decl, right_paren)
        return None

    @backtrack
    def _parse_declarator_brackets(self, left):
        left_bracket = self._parse_sign('[')
        if left_bracket is not None:
            if not self.has_more:
                self._raise_syntax_error("Expected ']'")
            constant = self._parse_constant_expression()
            if constant is None:
                right_bracket = self._parse_sign(']')
            else:
                right_bracket = self._expect_sign(']')
            if right_bracket is not None:
                return SubscriptExpr(left,
                                     left_bracket, constant, right_bracket)
        return None

    def _parse_direct_abstract_declarator(self):
        left = self._parse_declarator_parens(True)
        if left is None:
            left = self._parse_declarator_brackets(None)
            if left is None:
                return None
        while True:
            decl = self._parse_declarator_parens(True)
            if decl is None:
                params = self._parse_parameter_type_list()
                if params is None:
                    return left
                return FunctionExpr(left, params)
            left = decl

    def _parse_direct_declarator(self, abstract=False):
        """
        Returns a declarator or None
        """
        if abstract:
            return self._parse_direct_abstract_declarator()

        left = self._parse_declarator_parens()
        if left is None:
            left = self._parse_identifier()
        if left is None:
            return None

        while self.has_more:
            parameter_list = self._parse_parameter_type_list()
            if parameter_list is not None:
                left = FunctionExpr(left, parameter_list)
                continue
            brackets = self._parse_declarator_brackets(left)
            if brackets is not None:
                left = brackets
                continue
            break
        return left

    @backtrack
    def _parse_declarator(self, abstract=False):
        """
        Returns a declarator or None
        """
        star = self._parse_sign('*')
        if star is None:
            return self._parse_direct_declarator(abstract)
        type_qualifiers = self._parse_type_qualifier_list()
        right = self._parse_declarator(abstract)
        if right is None and not abstract:
            return None
        return PointerExpr(star, right, type_qualifiers)

    def _parse_strings(self):
        strings = []
        while self.has_more:
            token = self._parse_token('string')
            if token is None:
                break
            strings.append(LiteralExpr(token))
        if len(strings) == 0:
            return None
        return StringsExpr(strings)

    def _parse_primary_token(self):
        if not self.has_more:
            return None
        begin = self.index
        token = self.next()
        if token.kind in 'integer float character'.split():
            return token
        self.index = begin
        return self._parse_identifier_token()

    @backtrack
    def _parse_paren_expression(self):
        left_paren = self._parse_sign('(')
        if left_paren is None:
            return None
        expr = self.parse_expression()
        if expr is None:
            return None
        return ParenExpr(left_paren, expr, self._expect_sign(')'))

    def _parse_primary_expression(self):
        strings = self._parse_strings()
        if strings:
            return strings

        token = self._parse_primary_token()
        if token is not None:
            return LiteralExpr(token)
        return self._parse_paren_expression()

    def _parse_name_designator(self):
        dot = self._parse_sign('.')
        if dot is None:
            return None
        name = self._parse_identifier()
        if name is None:
            self._raise_syntax_error()
        return NameDesignatorExpr(dot, name)

    def _parse_designator(self):
        name_des = self._parse_name_designator()
        if name_des is not None:
            return name_des
        left_bracket = self._parse_sign('[')
        if left_bracket is None:
            return None
        index = self._parse_constant_expression()
        if index is None:
            self._raise_syntax_error()
        right_bracket = self._parse_sign(']')
        return IndexDesignatorExpr(left_bracket, index, right_bracket)

    def _parse_designation(self):
        designator = self._parse_designator()
        if designator is None:
            return None
        eq = self._expect_sign('=')
        right = self._parse_initializer()
        if right is None:
            self._raise_syntax_error()
        return BinaryOperationExpr(designator, eq, right)

    def _parse_initializer_list(self):
        left_brace = self._parse_sign('{')
        if left_brace is None:
            return None
        lst = []
        while True:
            if len(lst):
                comma = self._parse_sign(',')
                if comma is None:
                    break
                lst.append(comma)
            init = self._parse_designation()
            if init is None:
                init = self._parse_initializer()
                if init is None:
                    self._raise_syntax_error('Expected initializer')
            lst.append(init)
        right_brace = self._expect_sign('}')
        list_expr = CommaListExpr(lst)
        return InitializerListExpr(left_brace, list_expr, right_brace)

    def _parse_initializer(self):
        init_list = self._parse_initializer_list()
        if init_list is not None:
            return init_list
        expr = self._parse_assignment_expression()
        if expr is not None:
            return expr
        return None

    def _parse_init_declarator(self):
        """
        Returns a declarator or None
        """
        declarator = self._parse_declarator()
        equal = self._parse_sign('=')
        if declarator is not None and equal is not None:
            initializer = self._parse_initializer()
            if initializer is None:
                self._raise_syntax_error('Initializer expected')
            return BinaryOperationExpr(declarator, equal, initializer)
        return declarator

    def _parse_init_declarator_list(self):
        """
        Returns a CommaListExpr
        """
        declarators = []
        comma = None
        while self.has_more:
            declarator = self._parse_init_declarator()
            if declarator is None:
                if comma is None:
                    break
                else:
                    self._raise_syntax_error("Expected declarator after ','")
            declarators.append(declarator)
            comma = self._parse_sign(',')
            if comma is None:
                break
            declarators.append(comma)
        return CommaListExpr(declarators)

    def _parse_function_specifier(self):
        """
        Returns a TypeSpecifierExpr or None
        """
        kw_strings = 'inline noreturn'.split()
        token = self._parse_keyword(kw_strings)
        if token is None:
            return None
        return TypeSpecifierExpr(token)

    def _parse_storage_class_specifier(self):
        """
        Returns a TypeSpecifierExpr or None
        """
        kw_strings = 'typedef extern static auto register'.split()
        token = self._parse_keyword(kw_strings)
        if token is None:
            return None
        return TypeSpecifierExpr(token)

    def _get_type_specifiers_strings(self):
        return 'void char short int long float double signed unsigned'.split()

    def _parse_struct_or_union_specifier(self):
        kw = self._parse_keyword('struct union'.split())
        if kw is None:
            return None
        identifier = self._parse_identifier_token()
        compound = self._parse_compound_statement()
        if compound is not None:
            if len(compound.statements) > 0:
                self._raise_syntax_error('Expected type name')
        if identifier is None and compound is None:
            self._raise_syntax_error("Expected identifier or "
                                     "'{}' after {!r}".format('{', kw.string))
        return StructExpr(kw, identifier, compound)
        # TODO

    def _parse_enumerator(self):
        return self._parse_assignment_expression()

    def _parse_enumerator_list(self):
        left_brace = self._parse_sign('{')
        if left_brace is None:
            return None
        enums = []
        while True:
            enumerator = self._parse_enumerator()
            if enumerator is None:
                break
            enums.append(enumerator)
            comma = self._parse_sign(',')
            if comma is None:
                break
            enums.append(comma)
        right_brace = self._expect_sign('}')
        list_expr = CommaListExpr(enums, allow_trailing=True)
        return InitializerListExpr(left_brace, list_expr, right_brace)

    def _parse_enum_specifier(self):
        enum_keyword = self._parse_keyword(['enum'])
        if enum_keyword is None:
            return None
        identifier = self._parse_identifier_token()
        body = self._parse_enumerator_list()
        if identifier is None and body is None:
            self._raise_syntax_error("Expected identifier or '{' after 'enum'")
        return EnumExpr(enum_keyword, identifier, body)

    @backtrack
    def _parse_type_specifier(self, allowed_type_specs='bc'):
        """
        Returns an AbstractTypeSpecifierExpr or None
        """

        if 'b' in allowed_type_specs:
            kwd = self._parse_keyword(self._get_type_specifiers_strings())
            if kwd is not None:
                return TypeSpecifierExpr(kwd)

        if 'c' not in allowed_type_specs:
            return None

        struct = self._parse_struct_or_union_specifier()
        if struct is not None:
            return struct

        enum = self._parse_enum_specifier()
        if enum is not None:
            return enum

        token = self._parse_token('identifier')
        if token is not None and token.string in self.types:
            return TypeSpecifierExpr(token)
        return None

    def _parse_type_qualifier(self):
        """
        Returns a TypeSpecifierExpr or None
        """
        kw_strings = ('const volatile '.split())
        kw = self._parse_keyword(kw_strings)
        return None if kw is None else TypeSpecifierExpr(kw)

    def _parse_type_qualifier_list(self):
        """
        Returns a list of tokens
        """
        qualifiers = []
        while True:
            qualifier = self._parse_type_qualifier()
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
            spec = self._parse_storage_class_specifier()
            if spec is not None:
                specs.append(spec)

        spec = self._parse_type_specifier(allowed_type_specs)
        if spec is not None:
            specs.append(spec)
            is_keyword = (isinstance(spec, TypeSpecifierExpr) and
                          spec.token.kind == 'keyword')
            if is_keyword:
                allowed_type_specs = 'b'
            else:
                allowed_type_specs = ''

        spec = self._parse_type_qualifier()
        if spec is not None:
            specs.append(spec)

        spec = self._parse_function_specifier()
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

    def _parse_declaration_specifiers(self):
        """
        Returns a TypeExpr or None
        """
        return self._parse_type_specifiers(True)

    def _parse_specifier_qualifier_list(self):
        """
        Returns a TypeExpr or None
        """
        return self._parse_type_specifiers(False)

    def _parse_type_name(self):
        specifiers = self._parse_specifier_qualifier_list()
        if specifiers is None:
            return None
        decl = self._parse_declarator(abstract=True)
        return TypeNameExpr(specifiers, decl)

    def _filter_type_specifiers(self, tokens):
        return [t for t in tokens if
                t.string in self._get_type_specifiers_strings()]

    def _add_type_from_declarator(self, decl):
        name = get_declarator_name(decl)
        if name is None:
            msg = ('Cannot retrieve the name of the declarator '
                   '{!r} (repr: {!r})'.format(str(decl), decl))
            raise Exception(msg)
        self.add_type(name)

    @backtrack
    def parse_declaration(self):
        type_expr = self._parse_declaration_specifiers()
        if type_expr is None:
            return None
        if type_expr.is_typedef:
            self._in_typedef = True
        declarators = self._parse_init_declarator_list()
        self._in_typedef = False
        semicolon = self._parse_sign(';')
        if semicolon is None:
            return None
        if type_expr.is_typedef:
            if len(declarators) == 0:
                self._raise_syntax_error("Expected type name after 'typedef'")
            for decl in declarators.children:
                self._add_type_from_declarator(decl)
        return DeclarationExpr(type_expr, declarators, semicolon)

    def _parse_declaration_list(self):
        declarations = []
        while True:
            decl = self.parse_declaration()
            if decl is None:
                break
            declarations.append(decl)
        return declarations

    def _parse_argument_expression_list(self):
        args_commas = []
        while True:
            if len(args_commas) > 0:
                comma = self._parse_sign(',')
                if comma is None:
                    break
                args_commas.append(comma)
            argument = self._parse_assignment_expression()
            if argument is None:
                argument = self._parse_type_name()
                if argument is None:
                    if len(args_commas) == 0:
                        break
                    self._raise_syntax_error('Expected argument')
            args_commas.append(argument)
        return CommaListExpr(args_commas)

    @backtrack
    def _parse_compound_literal(self):
        parens = self._parse_parenthesed_type_name()
        if parens is None:
            return None
        compound = self._parse_initializer_list()
        if compound is None:
            return None
        return CompoundLiteralExpr(parens, compound)

    def _parse_postfix_expression(self):
        left = self._parse_primary_expression()
        if left is None:
            left = self._parse_compound_literal()
            if left is None:
                return None
        while True:
            operator = self._parse_sign('[ ( ++ -- . ->'.split())
            if operator is None:
                break
            if operator.string == '(':
                args_commas = self._parse_argument_expression_list()
                right_paren = self._expect_sign(')')
                left = CallExpr(left, operator, args_commas, right_paren)
            elif operator.string == '[':
                expr = self.parse_expression()
                right_bracket = self._expect_sign(']')
                left = SubscriptExpr(left, operator, expr, right_bracket)
            elif operator.string in '++ --'.split():
                left = UnaryOperationExpr(operator, left, postfix=True)
            elif operator.string in '. ->'.split():
                identifier = self._parse_identifier()
                if identifier is None:
                    self._raise_syntax_error('Expected an identifier')
                left = BinaryOperationExpr(left, operator, identifier)
            else:
                raise Exception()
        return left

    def _parse_unary_operator(self):
        """
        Returns a token or None
        """
        return self._parse_sign('& * + - ~ !'.split())

    def _parse_sizeof(self):
        sizeof = self._parse_keyword(['sizeof'])
        if sizeof is None:
            return None
        expr = self._parse_unary_expression()
        if expr is not None:
            return SizeofExpr(sizeof, expr)
        left_paren = self._expect_sign('(')
        type_name = self._parse_type_name()
        if type_name is None:
            self._raise_syntax_error('Expected type name')
        right_paren = self._expect_sign(')')
        return SizeofExpr(sizeof,
                          ParenExpr(left_paren, type_name, right_paren))

    def _parse_unary_expression(self):
        sizeof = self._parse_sizeof()
        if sizeof is not None:
            return sizeof
        operator = self._parse_unary_operator()
        if operator is None:
            operator = self._parse_sign('-- ++'.split())
            if operator is None:
                return self._parse_postfix_expression()
            expr = self._parse_unary_expression()
            return UnaryOperationExpr(operator, expr)
        expr = self._parse_cast_expression()
        return UnaryOperationExpr(operator, expr)

    def _parse_binary_operation(self, operators, sub_function):
        """
        operators: a string or a list of strings
        sub_function: a function
        """

        if isinstance(operators, str):
            operators = operators.split()
        left = sub_function()
        while True:
            operator = self._parse_sign(operators)
            if operator is None:
                break
            right = sub_function()
            if right is None:
                self._raise_syntax_error()
            left = BinaryOperationExpr(left, operator, right)
        return left

    @backtrack
    def _parse_parenthesed_type_name(self):
        left_paren = self._parse_sign('(')
        if left_paren is None:
            return None
        type_name = self._parse_type_name()
        if type_name is None:
            return None
        right_paren = self._expect_sign(')')
        if right_paren is None:
            return None
        return ParenExpr(left_paren, type_name, right_paren)

    def _parse_cast_expression(self):
        expr = self._parse_unary_expression()
        if expr is not None:
            return expr
        parens = self._parse_parenthesed_type_name()
        if parens is None:
            return None
        expr = self._parse_cast_expression()
        if expr is None:
            return self._raise_syntax_error('Expected expression')
        return CastExpr(parens, expr)

    def _parse_multiplicative_expression(self):
        return self._parse_binary_operation('* / %',
                                            self._parse_cast_expression)

    def _parse_additive_expression(self):
        f = self._parse_multiplicative_expression
        return self._parse_binary_operation('+ -', f)

    def _parse_shift_expression(self):
        return self._parse_binary_operation('>> <<',
                                            self._parse_additive_expression)

    def _parse_relational_expression(self):
        return self._parse_binary_operation('< > <= >=',
                                            self._parse_shift_expression)

    def _parse_equality_expression(self):
        return self._parse_binary_operation('== !=',
                                            self._parse_relational_expression)

    def _parse_and_expression(self):
        return self._parse_binary_operation('&',
                                            self._parse_equality_expression)

    def _parse_exclusive_or_expression(self):
        return self._parse_binary_operation('^',
                                            self._parse_and_expression)

    def _parse_inclusive_or_expression(self):
        func = self._parse_exclusive_or_expression
        return self._parse_binary_operation('|', func)

    def _parse_logical_and_expression(self):
        func = self._parse_inclusive_or_expression
        return self._parse_binary_operation('&&', func)

    def _parse_logical_or_expression(self):
        func = self._parse_logical_and_expression
        return self._parse_binary_operation('||', func)

    def _parse_conditional_expression(self):
        left = self._parse_logical_or_expression()
        quest = self._parse_sign('?')
        if quest is None:
            return left
        mid = self.parse_expression()
        if mid is None:
            self._raise_syntax_error('Expected expression')
        colon = self._expect_sign(':')
        right = self._parse_conditional_expression()
        return TernaryOperationExpr(left, quest, mid, colon, right)

    def _parse_constant_expression(self):
        return self._parse_conditional_expression()

    def _parse_assignment_operator(self):
        """
        Returns an assignment operator or None
        """
        ops = '= *= /= %= += -= <<= >>= &= ^= |='.split()
        return self._parse_sign(ops)

    @backtrack
    def _parse_assignment_expression_2(self):
        unary = self._parse_unary_expression()
        if unary is None:
            return None
        operator = self._parse_assignment_operator()
        if operator is None:
            return None
        right = self._parse_assignment_expression()
        return BinaryOperationExpr(unary, operator, right)

    def _parse_assignment_expression(self):
        begin = self.index
        left = self._parse_conditional_expression()
        if left is None:
            return None
        left_end = self.index
        self.index = begin
        assign = self._parse_assignment_expression_2()
        if assign is None:
            self.index = left_end
            return left
        return assign

    def parse_expression(self):
        return self._parse_assignment_expression()

    def _parse_expression_statement(self):
        semicolon = self._parse_sign(';')
        if semicolon is not None:
            return StatementExpr(None, semicolon)
        expr = self.parse_expression()
        if expr is None:
            return None
        return StatementExpr(expr, self._expect_sign(';'))

    def _parse_selection_statement(self):
        switch_token = self._parse_keyword(['switch'])
        if switch_token is not None:
            self._raise_syntax_error("The 'switch' statement is not "
                                     "implemented")

        if_token = self._parse_keyword(['if'])
        if if_token is None:
            return None
        left_paren = self._expect_sign('(')
        expr = self.parse_expression()
        if expr is None:
            self._raise_syntax_error('Expected expression')
        right_paren = self._expect_sign(')')
        statement = self.parse_statement()
        if statement is None:
            self._raise_syntax_error('Expected statement')

        else_statement = None
        else_token = self._parse_keyword(['else'])
        if else_token is not None:
            else_statement = self.parse_statement()
            if else_statement is None:
                self._raise_syntax_error("Expected statement after 'else'")
        return IfExpr(if_token,
                      ParenExpr(left_paren, expr, right_paren), statement,
                      else_token, else_statement)

    def _parse_iteration_statement(self):
        token = self._parse_keyword('do for'.split())
        if token is not None:
            self._raise_syntax_error("'do' and 'for' statements are not "
                                     "implemented")
        while_token = self._parse_keyword('while'.split())
        if while_token is None:
            return None
        left_paren = self._expect_sign('(')
        expr = self.parse_expression()
        if expr is None:
            self._raise_syntax_error('Expected expression')
        right_paren = self._expect_sign(')')
        statement = self.parse_statement()
        if statement is None:
            self._raise_syntax_error('Expected statement')
        return WhileExpr(while_token,
                         ParenExpr(left_paren, expr, right_paren),
                         statement)

    def _parse_return_statement(self):
        return_token = self._parse_keyword(['return'])
        if return_token is None:
            return None
        expr = None
        semicolon = self._parse_sign(';')
        if semicolon is None:
            expr = self.parse_expression()
            if expr is None:
                self._raise_syntax_error('Expected expression')
            semicolon = self._expect_sign(';')
        return StatementExpr(JumpExpr(return_token, expr), semicolon)

    def _parse_jump_statement(self):
        r = self._parse_return_statement()
        if r is not None:
            return r
        token = self._parse_keyword(['break', 'continue'])
        if token is None:
            return None
        semicolon = self._expect_sign(';')
        return StatementExpr(JumpExpr(token, None), semicolon)

    def parse_statement(self):
        stmt = self._parse_compound_statement()
        if stmt is not None:
            return stmt
        stmt = self._parse_expression_statement()
        if stmt is not None:
            return stmt
        stmt = self._parse_selection_statement()
        if stmt is not None:
            return stmt
        stmt = self._parse_iteration_statement()
        if stmt is not None:
            return stmt
        stmt = self._parse_jump_statement()
        if stmt is not None:
            return stmt
        return None

    def _parse_statement_list(self):
        statements = []
        while True:
            statement = self.parse_statement()
            if statement is None:
                break
            statements.append(statement)
        return statements

    def _parse_compound_statement(self):
        left_brace = self._parse_sign('{')
        if left_brace is None:
            return None
        declarations = self._parse_declaration_list()
        statements = self._parse_statement_list()
        right_brace = self._expect_sign('}')
        return CompoundExpr(left_brace, declarations, statements, right_brace)

    def _parse_function_definition(self):
        type_expr = self._parse_declaration_specifiers()
        declarator = self._parse_declarator()
        if declarator is None:
            if type_expr is None:
                return None
            self._raise_syntax_error('Expected declarator')
        compound = self._parse_compound_statement()
        if compound is None:
            self._raise_syntax_error('Expected compound statement')
        return FunctionDefinitionExpr(type_expr, declarator, compound)

    def _parse_external_declaration(self):
        decl = self.parse_declaration()
        if decl is not None:
            return decl
        return self._parse_function_definition()

    def _parse_translation_unit(self):
        declarations = []
        while self.has_more:
            decl = self._parse_external_declaration()
            if decl is None and self.has_more:
                s = 'Unexpected {!r}'.format(self.next().string)
                self._raise_syntax_error(s)
            if decl is not None:
                declarations.append(decl)
        expr = TranslationUnitExpr(declarations)
        self._expand_directives()
        if len(self._directives) > 0:
            raise Exception('Parser internal error')
        return expr

    def parse(self):
        return self._parse_translation_unit()


def convert_to_pp_result(v, file_name='<unknown file>', include_dirs=None):
    if isinstance(v, str):
        tokens = lex(v, file_name)
        return convert_to_pp_result(tokens, file_name, include_dirs)
    if isinstance(v, list):
        return preprocess(v, include_dirs)
    if not isinstance(v, PreprocessorResult):
        raise ValueError('Expected a PreprocessorResult')
    return v


def parse(v,
          file_name='<unknown file>',
          include_directories=None,
          included_file_cache=None):
    """
    v: a string or a preprocessor result

    Return a tuple (expr, issues)
    """
    assert isinstance(file_name, str)

    if include_directories is None:
        include_directories = []

    pp_result = convert_to_pp_result(v, file_name, include_directories)
    parser = Parser(pp_result, file_name,
                    include_directories,
                    included_file_cache)
    tree = parser.parse()
    assert isinstance(tree, Expr)
    """
    tokens = preprocess(tokens, include_directories).tokens
    if tokens != tree.tokens:
        if isinstance(v, str):
            print(v)
        print(tree)
        print()
        print('\n'.join(repr(t) for t in tokens))
        print()
        print('\n'.join(repr(t) for t in tree.tokens))
    assert tokens == tree.tokens
    """
    return tree, parser.issues


def parse_statement(v):
    """
    v: a string or a list of tokens

    Return a tuple (expr, issues)
    """
    parser = Parser(convert_to_pp_result(v))
    return parser.parse_statement(), parser.issues


def parse_expr(v):
    """
    v: a string or a list of tokens

    Return a tuple (expr, issues). 'expr' can be None.
    """
    parser = Parser(convert_to_pp_result(v))
    return parser.parse_expression(), parser.issues


class StyleChecker:
    DEFAULT_TAB_WIDTH = 8

    def __init__(self, issue_handler):
        self._issue_handler = issue_handler
        self.tab_width = StyleChecker.DEFAULT_TAB_WIDTH
        self.indent_chars = ' \t'

    @staticmethod
    def get_file_name(source_tokens):
        """
        Return the name of the file where the tokens arise from.
        """
        if len(source_tokens) == 0:
            return '<unknown file>'
        return source_tokens[0].begin.file_name

    def create_argument_group(self, parser):
        pass

    def configure_all(self, options):
        self.tab_width = options.width
        self.configure(options)

    def configure(self, options):
        pass

    def issue(self, issue):
        assert isinstance(issue, StyleIssue)
        self._issue_handler(issue)

    def warn(self, message, position):
        self.issue(StyleIssue(message, position, level='warn'))

    def error(self, message, position):
        self.issue(StyleIssue(message, position))

    def check(self, source_tokens, pp_tokens, expr):
        raise NotImplementedError('Not implemented in {}'.format(
            type(self).__name__))

    def check_source(self, source, source_tokens, pp_tokens, expr):
        """
        Like a StyleChecker, but accepts a string containing the source
        code of the file
        """
        # pylint: disable=unused-argument
        return self.check(source_tokens, pp_tokens, expr)

    def check_same_line(self, token_a, token_b):
        """
        Check if the two given tokens are on the same line
        """
        assert isinstance(token_a, Token)
        assert isinstance(token_b, Token)

        if token_a.end.line != token_b.begin.line:
            msg = '{!r} is not on the same line than {!r}'.format(
                token_a.string, token_b.string)
            self.error(msg, token_a.end)

    @staticmethod
    def get_previous_token(tokens, i):
        return None if i == 0 else tokens[i - 1]

    @staticmethod
    def get_next_token(tokens, i):
        return None if i == len(tokens) - 1 else tokens[i + 1]

    @staticmethod
    def _pop_char(string):
        if len(string) == 0:
            raise Exception()
        return string[0], string[1:]

    def get_visible_indent_width(self, indent_string,
                                 position, visible_column=1):
        """
        Returns the visible width in the given string.

        indent_string: A string constitued of spaces and tabs.
        position: Only used to raise some errors
        """
        column = visible_column
        begin_column = column
        tabs = 0
        spaces = 0
        while len(indent_string) > 0:
            char, indent_string = self._pop_char(indent_string)
            if char not in self.indent_chars:
                raise Exception()
            if char == '\t':
                if spaces > 0:
                    self.error('Unexpected tabulation after spaces', position)
                tabs += 1
                tab_visible_width = (self.tab_width -
                                     (column - 1) % self.tab_width)
                column += tab_visible_width
            elif char == ' ':
                spaces += 1
                column += 1
            else:
                raise Exception()
        return column - begin_column

    def split_indent_string(self, string):
        indent_string = ''
        while len(string) > 0 and string[0] in self.indent_chars:
            indent_string += string[0]
            string = string[1:]
        return indent_string, string

    def get_visible_width(self, string, position, visible_column=1):
        """
        position: Only used to raise some errors
        """
        if len(string) == 0:
            return 0

        indent_string, string = self.split_indent_string(string)
        width = self.get_visible_indent_width(indent_string,
                                              position,
                                              visible_column)

        while len(string) > 0 and string[0] not in self.indent_chars:
            width += 1
            string = string[1:]
        visible_column += width
        return width + self.get_visible_width(string, position,
                                              visible_column)

    def check_indent(self, string, expected_width, position,
                     visible_column=1):
        indent_string, string = self.split_indent_string(string)

        width = self.get_visible_indent_width(indent_string, position,
                                              visible_column)
        if width != expected_width:
            diff = expected_width - width
            msg = 'Bad indent level, expected {} {} space{}'.format(
                abs(diff),
                'more' if diff > 0 else 'fewer',
                's' if abs(diff) > 1 else '')
            self.error(msg, position)

    def check_margin(self, source, left_token, margin, right_token):
        """
        Checks the margin between two tokens.

        If the given margin is a space, one space is expected.
        If the given margin is an int, this number is the length of
        the expected margin - it can be a mix of spaces an tabs.

        If the given tokens are on different lines, no check is performed.
        """
        one_space = (margin == ' ')

        left_end = left_token.end
        right_begin = right_token.begin

        if left_end.line != right_begin.line:
            return

        margin_source = source[left_end.index:right_begin.index]

        if one_space:
            if margin_source != ' ':
                msg = 'Expected one space between {!r} and {!r}'.format(
                    left_token.string, right_token.string)
                self.error(msg, left_end)
            return

        assert(isinstance(margin, int))
        visible_margin = self.get_visible_width(margin_source, left_end)
        if visible_margin != margin:
            msg = 'Expected {!r} spaces or tabs between {!r} and {!r}'.format(
                margin, left_token.string, right_token.string)
            self.error(msg, left_end)


class LineChecker(StyleChecker):
    def __init__(self, issue_handler):
        super().__init__(issue_handler)

    def check_line(self, begin, line, end):
        """
        This function is called on each line of the file.

        begin: The position of the first character of the line.
        line: The string of the line.
        end: The position of the last character of the line.
        """
        raise NotImplementedError()

    def check_source(self, source, source_tokens, pp_tokens, expr):
        lines = source.splitlines(True)
        index = 0
        file_name = self.get_file_name(source_tokens)
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
    def __init__(self, issue_handler):
        super().__init__(issue_handler)

    def check_line(self, begin, line, end):
        if len(line) > 80:
            self.error("Too long line (more than 80 characters)", end)


class EmptyLineChecker(LineChecker):
    def __init__(self, issue_handler):
        super().__init__(issue_handler)
        self._empty_previous_line = False

    def check_line(self, begin, line, end):
        empty = len(line.strip()) == 0
        if empty:
            if self._empty_previous_line:
                self.error("Empty line", begin)
        self._empty_previous_line = empty

    def check_source(self, source, source_tokens, pp_tokens, expr):
        self._empty_previous_line = False
        super().check_source(source, source_tokens, pp_tokens, expr)


class EmptyLineInFunctionChecker(StyleChecker):
    def __init__(self, issue_handler):
        super().__init__(issue_handler)

    def _check_block(self, expr_list):
        tokens = []
        for expr in expr_list:
            tokens += expr.tokens
        line = tokens[0].end.line
        for token in tokens:
            if token.end.line > line + 1:
                self.error('Unexpected empty line', token.begin)
            line = token.end.line

    def _check_compound(self, compound):
        if len(compound.declarations) > 0:
            self._check_block(compound.declarations)
        if len(compound.statements) > 0:
            self._check_block(compound.statements)
        if len(compound.declarations) > 0 and len(compound.statements) > 0:
            last_decl = compound.declarations[-1]
            first_stmt = compound.statements[0]
            decl_end = last_decl.last_token.end
            stmt_begin = first_stmt.first_token.begin
            if decl_end.line + 1 == stmt_begin.line:
                self.error('Expected empty line between declarations and '
                           'statements', decl_end)

    def check(self, source_tokens, pp_tokens, expr):
        for function in expr.select('function_definition'):
            pass
        for compound in expr.select('compound'):
            self._check_compound(compound)


class SupinfoChecker(LineChecker):
    def __init__(self, issue_handler):
        super().__init__(issue_handler)

    def check_line(self, begin, line, end):
        if 'supinfo' in line.lower():
            self.warn("Advertisment for another school", begin)


class TrailingWhitespaceChecker(LineChecker):
    def __init__(self, issue_handler):
        super().__init__(issue_handler)

    def check_line(self, begin, line, end):
        if line.rstrip() != line:
            self.warn("Trailing whitespaces at the end of the line", end)


class HeaderCommentChecker(StyleChecker):
    def __init__(self, issue_handler):
        self.username_check_enabled = False
        super().__init__(issue_handler)

    def configure(self, options):
        self.username_check_enabled = options.header_username

    def create_argument_group(self, parser):
        group = parser.add_argument_group('Header comments')
        group.add_argument('--header-username',
                           action='store_true',
                           help="check the username in header comments")
        return group

    def _check_username(self, login, position):
        if not self.username_check_enabled:
            return
        if not re.match(r'\w+_\w', login):
            self.warn("Not a valid EPITECH username (was {!r})".format(login),
                      position)

    def check_line(self, i, line, position):
        if i == 3:
            login = line.split()[-1]
            self._check_username(login, position)
            if not line.startswith("** Made by"):
                self.error("Invalid 'Made by' line", position)
        if i == 6:
            login = line.split()[-1]
            self._check_username(login, position)
            if not line.startswith("** Started on"):
                self.error("Invalid 'Started on' line", position)
        if i == 7:
            login = line.split()[-1]
            self._check_username(login, position)
            if not line.startswith("** Last update"):
                self.error("Invalid 'Last update' line", position)

    def _check_header(self, token):
        lines = token.string.splitlines()
        if len(lines) != 9:
            self.error('The header must be 9 lines long', token.begin)
        for i, line in enumerate(lines):
            self.check_line(i, line, token.begin)

    def check(self, source_tokens, pp_tokens, expr):
        for token in source_tokens:
            if token.kind == 'comment' and token.begin.line == 1:
                self._check_header(token)
                return
        self.error("No header comment", source_tokens[0].begin)


class CommentChecker(StyleChecker):
    def __init__(self, issue_handler):
        super().__init__(issue_handler)

    def _get_all_tokens_in_expr(self, tokens, expr):
        first = expr.first_token
        last = expr.last_token
        all_tokens = tokens[tokens.index(first):tokens.index(last) + 1]
        assert first == all_tokens[0]
        assert last == all_tokens[-1]
        return all_tokens

    def _check_comment(self, comment):
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
            if line[:2] != '**':
                self.error("The comment lines should start with '**'",
                           comment.begin)
            if line[2:3].strip() != '':
                self.error("Expected a space after '**'",
                           comment.begin)

    def check(self, source_tokens, pp_tokens, expr):
        for token in source_tokens:
            if token.kind == 'comment':
                self._check_comment(token)

        funcs = expr.select('function_definition')
        for func in funcs:
            func_tokens = self._get_all_tokens_in_expr(source_tokens, func)
            for token in func_tokens:
                if token.kind == 'comment':
                    self.error('Comment inside a function', token.begin)


class BinaryOpSpaceChecker(StyleChecker):
    def __init__(self, issue_handler):
        super().__init__(issue_handler)

    def check_source(self, source, source_tokens, pp_tokens, expr):
        bin_ops = expr.select('binary_operation')
        for operation in bin_ops:
            left_token = operation.left.last_token
            right_token = operation.right.first_token
            operator = operation.operator
            margin = 0 if operator.string in '. ->'.split() else ' '
            self.check_margin(source, left_token, margin, operator)
            self.check_margin(source, operator, margin, right_token)


class UnaryOpSpaceChecker(StyleChecker):
    def __init__(self, issue_handler):
        super().__init__(issue_handler)

    def check_source(self, source, source_tokens, pp_tokens, expr):
        unary_ops = expr.select('unary_operation')
        for operation in unary_ops:
            if operation.postfix:
                left_token = operation.right.last_token
                operator = operation.operator
                self.check_margin(source, left_token, 0, operator)
            else:
                right_token = operation.right.first_token
                operator = operation.operator
                self.check_margin(source, operator, 0, right_token)


class ReturnChecker(StyleChecker):
    """
    Check for parentheses after 'return'.
    """

    def __init__(self, issue_handler):
        super().__init__(issue_handler)

    def _check_return(self, statement_expr):
        return_expr = statement_expr.expression
        if return_expr.expression is None:
            return
        if not isinstance(return_expr.expression, ParenExpr):
            self.error("Missing parentheses after 'return'",
                       return_expr.keyword.end)
            return
        left_paren = return_expr.expression.left_paren
        self.check_same_line(return_expr.keyword, left_paren)

    def check(self, source_tokens, pp_tokens, expr):
        for statement in expr.select('statement'):
            if (isinstance(statement.expression, JumpExpr) and
                    statement.expression.keyword.string == 'return'):
                self._check_return(statement)


class KeywordSpaceChecker(StyleChecker):
    def __init__(self, issue_handler):
        super().__init__(issue_handler)

    def check_source(self, source, source_tokens, pp_tokens, expr):
        need_one_space = 'break continue if return while'.split()
        for i, token in enumerate(pp_tokens):
            if token.kind != 'keyword':
                continue
            next_token = self.get_next_token(pp_tokens, i)
            if token.string in need_one_space:
                self.check_margin(source, token, ' ', next_token)


class ParenChecker(StyleChecker):
    def __init__(self, issue_handler):
        super().__init__(issue_handler)

    def _check_paren_expr(self, source, paren):
        if isinstance(paren, CallExpr):
            inner = paren.arguments
        else:
            inner = paren.expression
        if len(inner.tokens) == 0:
            self.check_margin(source, paren.left_paren, 0, paren.right_paren)
            return
        self.check_margin(source, paren.left_paren, 0, inner.first_token)
        self.check_margin(source, inner.last_token, 0, paren.right_paren)

    def check_source(self, source, source_tokens, pp_tokens, expr):
        for paren in expr.select('paren'):
            self._check_paren_expr(source, paren)
        for call in expr.select('call'):
            self._check_paren_expr(source, call)


class FunctionLengthChecker(StyleChecker):
    def __init__(self, issue_handler):
        super().__init__(issue_handler)

    def check(self, source_tokens, pp_tokens, expr):
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
    def __init__(self, issue_handler):
        super().__init__(issue_handler)

    def check(self, source_tokens, pp_tokens, expr):
        funcs = expr.select('function_definition')
        if len(funcs) > 5:
            self.error('Too many functions in a file (more than 5)',
                       expr.first_token.begin)


class DirectiveIndentationChecker(StyleChecker):
    def __init__(self, issue_handler):
        super().__init__(issue_handler)

    def check(self, source_tokens, pp_tokens, expr):
        level = 0
        for token in source_tokens:
            if token.kind != 'directive':
                continue

            string = token.string.strip()[1:]
            name = string.strip()
            if name.startswith('endif'):
                if level == 0:
                    self.warn("Unexpected '#endif'", token.begin)
                else:
                    level -= 1
            local_level = level
            if name.startswith('else') or name.startswith('elif'):
                local_level -= 1
            self.check_indent(string, local_level, token.begin,
                              token.begin.column + 1)
            if name.startswith('if'):
                level += 1


class DeclarationChecker(StyleChecker):
    def __init__(self, issue_handler):
        super().__init__(issue_handler)

    def _check_struct(self, struct):
        if struct.identifier is not None:
            self.check_same_line(struct.struct, struct.identifier)

    def _check_declaration(self, decl):
        if len(decl.declarators.children) > 0:
            self.check_same_line(decl.declarators.last_token,
                                 decl.semicolon)

    def _check_function_def(self, func_def):
        if func_def.type_expr is not None:
            self.check_same_line(func_def.type_expr.last_token,
                                 func_def.declarator.first_token)

    def _check_new_line_constistency(self, expr):
        children = expr.children
        prev = None
        for child in children:
            if prev is not None and len(prev.tokens) and len(child.tokens):
                self.check_same_line(prev.last_token, child.first_token)
            prev = child

    def check(self, source_tokens, pp_tokens, expr):
        for struct in expr.select('struct'):
            self._check_struct(struct)
        for decl in expr.select('declaration'):
            self._check_declaration(decl)
            self._check_new_line_constistency(decl)
        for type_expr in expr.select('type'):
            self._check_new_line_constistency(type_expr)
        for func in expr.select('function_definition'):
            self._check_function_def(func)
        for decl in expr.select('function'):
            self._check_new_line_constistency(decl)


class DeclaratorChecker(StyleChecker):
    def __init__(self, issue_handler):
        super().__init__(issue_handler)

    def _check_function(self, source, function):
        self.check_margin(source,
                          function.declarator.last_token,
                          0,
                          function.parameters.first_token)

    def check_source(self, source, source_tokens, pp_tokens, expr):
        for func in expr.select('function'):
            self._check_function(source, func)


class CallChecker(StyleChecker):
    def __init__(self, issue_handler):
        super().__init__(issue_handler)

    def _check_call(self, source, call):
        self.check_margin(source,
                          call.expression.last_token,
                          0,
                          call.left_paren)

    def check_source(self, source, source_tokens, pp_tokens, expr):
        for func in expr.select('call'):
            self._check_call(source, func)


class OneStatementByLineChecker(StyleChecker):
    def __init__(self, issue_handler):
        super().__init__(issue_handler)

    def _check_statements(self, a, b):
        line_a = a.last_token.end.line
        line_b = b.first_token.begin.line
        if line_a == line_b:
            self.error('Multiple statements on the same line',
                       a.first_token.begin)

    def _check_children(self, expr):
        if isinstance(expr, (CompoundExpr, TranslationUnitExpr)):
            previous = None
            for child in expr.children:
                if previous is not None:
                    self._check_statements(previous, child)
                previous = child
        for child in expr.children:
            self._check_children(child)

    def check(self, source_tokens, pp_tokens, expr):
        self._check_children(expr)


class IndentationChecker(StyleChecker):
    def __init__(self, issue_handler):
        super().__init__(issue_handler)
        self.level = 0

    def _check_begin_indentation(self, lines, expr):
        if isinstance(expr, Expr):
            token = expr.first_token
        else:
            token = expr
        begin = token.begin
        first_line = lines[begin.line - 1]
        self.check_indent(first_line, self.level, begin, 1)

    def _check_end_indentation(self, lines, expr):
        end = expr.last_token.end
        first_line = lines[end.line - 1]
        self.check_indent(first_line, self.level, end, 1)

    def _check_expr(self, lines, expr):
        indented_classes = (
            FunctionDefinitionExpr,
            CompoundExpr,
            DeclarationExpr,
            StatementExpr,
            IfExpr,
            WhileExpr,
        )
        indentor_classes = (
            CompoundExpr,
            IfExpr,
            WhileExpr,
        )

        if isinstance(expr, indented_classes):
            self._check_begin_indentation(lines, expr)

        if isinstance(expr, indentor_classes):
            self.level += 2

        if isinstance(expr, IfExpr):
            self._check_expr(lines, expr.expression)
            self._check_expr(lines, expr.statement)
            if expr.else_statement is not None:
                self.level -= 2
                self._check_begin_indentation(lines, expr.else_token)
                self.level += 2
                if isinstance(expr.else_statement, IfExpr):
                    self.level -= 2
                self._check_expr(lines, expr.else_statement)
                if isinstance(expr.else_statement, IfExpr):
                    self.level += 2
        else:
            for child in expr.children:
                self._check_expr(lines, child)

        if isinstance(expr, indentor_classes):
            self.level -= 2

        if isinstance(expr, CompoundExpr):
            self._check_end_indentation(lines, expr)

    def check_source(self, source, source_tokens, pp_tokens, expr):
        self.level = 0
        self._check_expr(source.splitlines(), expr)
        if self.level != 0:
            self.warn('The indentation seems inconsistent',
                      source_tokens[0].begin)


class BraceChecker(StyleChecker):
    """
    Check compounds expressions only, don't check initializer lists.
    """

    def __init__(self, issue_handler):
        super().__init__(issue_handler)

    @staticmethod
    def _get_tokens_at_line(tokens, line):
        result = []
        for token in tokens:
            if token.begin.line == line or token.end.line == line:
                result.append(token)
        return result

    def _check_alone_in_line(self, tokens, token):
        tokens = BraceChecker._get_tokens_at_line(tokens, token.begin.line)
        assert token in tokens
        if len(tokens) > 1:
            self.error('{!r} not alone in its line'.format(token.string),
                       token.begin)

    def check(self, source_tokens, pp_tokens, expr):
        struct_compounds = expr.select('struct compound')
        for compound in expr.select('compound'):
            if compound in struct_compounds:
                continue
            self._check_alone_in_line(pp_tokens, compound.left_brace)
            self._check_alone_in_line(pp_tokens, compound.right_brace)


class CommaChecker(StyleChecker):
    def __init__(self, issue_handler):
        super().__init__(issue_handler)

    def _get_previous_and_next(self, tokens, i):
        return (self.get_previous_token(tokens, i),
                self.get_next_token(tokens, i))

    def _check_semicolon(self, source, tokens, i, token):
        prev_token, next_token = self._get_previous_and_next(tokens, i)
        if prev_token is None:
            self.error("Unexpected ';'", token.begin)
            return
        self.check_same_line(prev_token, token)
        if prev_token.string not in 'break continue return'.split():
            # There is an exception for the 'return' statement,
            # but this is checked in the ReturnChecker rather than
            # here.
            # There is also an exception for 'break' and 'continue'
            # statements.
            self.check_margin(source, prev_token, 0, token)

        if next_token is None:
            return
        if token.end.line == next_token.begin.line:
            self.error("{!r} on the same line than the previous ';'".format(
                next_token.string), next_token.begin)

    def _check_comma(self, source, tokens, i, token):
        prev_token, next_token = self._get_previous_and_next(tokens, i)
        assert prev_token is not None
        assert next_token is not None
        self.check_same_line(prev_token, token)
        self.check_margin(source, prev_token, 0, token)
        self.check_margin(source, token, ' ', next_token)

    def check_source(self, source, source_tokens, pp_tokens, expr):
        for i, token in enumerate(pp_tokens):
            if token.string == ',':
                self._check_comma(source, pp_tokens, i, token)
            elif token.string == ';':
                self._check_semicolon(source, pp_tokens, i, token)


class HeaderChecker(StyleChecker):
    def __init__(self, issue_handler):
        super().__init__(issue_handler)

    def _check_declarator(self, declarator):
        if isinstance(declarator, (CommaListExpr, PointerExpr)):
            for child in declarator.children:
                self._check_declarator(child)
            return
        elif isinstance(declarator, FunctionExpr):
            return
        self.error('This declaration is forbidden in a header file',
                   declarator.first_token.begin)

    def _check_declaration(self, declaration):
        type_expr = declaration.type_expr
        if type_expr.is_typedef:
            return
        elif type_expr.is_extern:
            self.warn('Global variable declaration',
                      declaration.first_token.begin)
            return
        self._check_declarator(declaration.declarators)

    @staticmethod
    def _remove_leading_hash(directive):
        assert directive.kind == 'directive'
        string = directive.string.strip()
        assert string[0] == '#'
        return string[1:].strip()

    def _get_header_name_h_macro(self, tokens):
        file_name = self.get_file_name(tokens)
        assert file_name.endswith('.h')
        file_name = file_name[:-2]
        if '/' in file_name:
            file_name = file_name[file_name.rindex('/') + 1:]
        return file_name.upper() + '_H_'

    def _get_ifndef_guard(self, tokens):
        return 'ifndef ' + self._get_header_name_h_macro(tokens)

    def _get_define_guard(self, tokens):
        return 'define ' + self._get_header_name_h_macro(tokens)

    def _get_endif_guard(self, tokens):
        return 'endif /* !' + self._get_header_name_h_macro(tokens) + ' */'

    def _check_directive(self, token, expected_string):
        if token.kind != 'directive':
            self.error('Expected the include guard directive {!r}'.format(
                '#' + expected_string), token.begin)
            return
        string = self._remove_leading_hash(token)
        if expected_string != string:
            msg = 'Bad once include guard directive (expected {!r})'.format(
                '#' + expected_string)
            self.error(msg, token.begin)

    def _check_once_include_guard(self, tokens):
        if len(tokens) < 2:
            self.error('Missing once include guard', tokens[0].begin)
            return
        self._check_directive(tokens[1], self._get_ifndef_guard(tokens))
        self._check_directive(tokens[2], self._get_define_guard(tokens))
        self._check_directive(tokens[-1], self._get_endif_guard(tokens))

    def check(self, source_tokens, pp_tokens, expr):
        if not self.get_file_name(source_tokens).endswith('.h'):
            return
        self._check_once_include_guard(source_tokens)
        for child in expr.children:
            if not isinstance(child, DeclarationExpr):
                self.error('This is forbidden in a header file',
                           child.first_token.begin)
                continue
            self._check_declaration(child)


class SourceFileChecker(StyleChecker):
    def __init__(self, issue_handler):
        super().__init__(issue_handler)

    def _check_directive(self, directive):
        string = directive.string[1:].strip()
        name = string.split()[0]
        if name in ('define', 'undef'):
            self.warn('The most of the {!r} directives are forbidden '
                      'in source files'.format('#' + name),
                      directive.begin)

    def _check_declaration(self, declaration):
        self.warn('Declaration in source file', declaration.first_token.begin)

    def check(self, source_tokens, pp_tokens, expr):
        if not self.get_file_name(source_tokens).endswith('.c'):
            return
        for token in source_tokens:
            if token.kind == 'directive':
                self._check_directive(token)
        for child in expr.children:
            if isinstance(child, DeclarationExpr):
                self._check_declaration(child)


class NameChecker(StyleChecker):
    def __init__(self, issue_handler):
        super().__init__(issue_handler)
        self._identifier_re = re.compile(r"^[a-z0-9_]*\Z", re.ASCII)
        self._macro_re = re.compile(r"^[A-Z0-9_]*\Z", re.ASCII)

    def _is_valid_lowercase_name(self, name):
        return re.match(self._identifier_re, name)

    def _is_valid_macro_name(self, name):
        return re.match(self._macro_re, name)

    def _check_lowercase_name(self, name, position):
        if not self._is_valid_lowercase_name(name):
            self.error('{!r} is an invalid name'.format(name), position)

    def _check_macro_name(self, name, position):
        if not self._is_valid_macro_name(name):
            self.error('{!r} is an invalid macro name'.format(name), position)

    def _check_lowercase_token(self, token):
        assert token.kind == 'identifier'
        self._check_lowercase_name(token.string, token.begin)

    def _check_declarator(self, decl, typedef=False, global_variable=False):
        if isinstance(decl, LiteralExpr) and decl.kind == 'identifier':
            self._check_lowercase_token(decl.token)
            if typedef and not decl.string.startswith('t_'):
                self.error('Invalid type name', decl.token.begin)
            elif global_variable and not decl.string.startswith('g_'):
                self.error('Invalid global variable name', decl.token.begin)
            return
        if isinstance(decl, SubscriptExpr):
            self._check_declarator(decl.expression)
            return
        if isinstance(decl, FunctionExpr):
            return
        if isinstance(decl, BinaryOperationExpr):
            if decl.operator.string == '=':
                # Don't check the initializer
                self._check_declarator(decl.left, typedef, global_variable)
                return
        for child in decl.children:
            self._check_declarator(child, typedef, global_variable)

    def _check_struct(self, struct):
        if struct.compound is None:
            return
        prefix_map = {
            'struct': 's_',
            'union': 'u_',
        }
        if struct.identifier is not None:
            self._check_lowercase_token(struct.identifier)
            prefix = prefix_map[struct.kind]
            if not struct.identifier.string.startswith(prefix):
                self.error('Invalid {} name'.format(struct.kind),
                           struct.identifier.begin)

    def _check_declaration(self, declaration, global_variable=False):
        typedef = declaration.type_expr.is_typedef
        if typedef:
            global_variable = False
        for declarator in declaration.declarators.children:
            self._check_declarator(declarator,
                                   typedef=typedef,
                                   global_variable=global_variable)

    def _check_file_name(self, tokens):
        file_name = self.get_file_name(tokens)
        cleaned = os.path.basename(file_name)
        # Remove the '.' of the extension
        cleaned = cleaned.replace('.', '')
        self._check_lowercase_name(cleaned, Position(file_name))

    def check(self, source_tokens, pp_tokens, expr):
        self._check_file_name(source_tokens)

        global_exprs = expr.children
        for decl in expr.select('declaration'):
            self._check_declaration(decl, decl in global_exprs)

        for func in expr.select('function'):
            self._check_declarator(func.declarator)

        for param in expr.select('parameter'):
            if param.declarator is not None:
                self._check_declarator(param.declarator)

        for struct in expr.select('struct'):
            self._check_struct(struct)

        for token in source_tokens:
            if token.kind != 'directive':
                continue
            string = token.string[1:].strip()
            if not string.startswith('define'):
                continue
            name = string.split()[1]
            if '(' in name:
                name = name[:name.index('(')]
            self._check_macro_name(name, token.begin)


class DeclaratorAlignmentChecker(StyleChecker):
    def __init__(self, issue_handler):
        super().__init__(issue_handler)

    def _get_token_indent(self, lines, token):
        begin = token.begin
        line = lines[begin.line - 1]
        before = line[:begin.column - 1]
        return self.get_visible_width(before, token.begin)

    def _get_declarator_indent(self, lines, declaration):
        declarators = declaration.declarators
        if len(declarators.children) == 0:
            return -1
        return self._get_token_indent(lines, declarators.first_token)

    def _check_declaration(self, lines, declaration, expected_indent=-1):
        indent = self._get_declarator_indent(lines, declaration)
        if indent == -1:
            return
        declarator_begin = declaration.declarators.first_token.begin
        if indent % self.tab_width != 0:
            self.error('Misaligned declarator', declarator_begin)
            return
        if expected_indent != -1 and expected_indent != indent:
            msg = 'Misaligned declarator (expected at the column {})'.format(
                expected_indent)
            self.error(msg, declarator_begin)

    def _check_compound(self, lines, compound, expected_indent=-1):
        decls = compound.declarations
        if len(decls) == 0:
            return
        if expected_indent == -1:
            expected_indent = self._get_declarator_indent(lines, decls[0])
            if expected_indent == -1:
                return
        for declaration in decls:
            self._check_declaration(lines, declaration, expected_indent)

    def _check_struct(self, lines, struct):
        assert struct.compound is not None
        indent = -1
        if struct.identifier is not None:
            indent = self._get_token_indent(lines, struct.identifier)
            if indent % self.tab_width != 0:
                self.error('Misaligned {} name'.format(struct.struct),
                           struct.identifier.begin)
        self._check_compound(lines, struct.compound, indent)

    def _check_function_def(self, lines, function):
        indent = self._get_token_indent(lines,
                                        function.declarator.first_token)
        if indent % self.tab_width != 0:
            self.error('Misaligned function name',
                       function.declarator.first_token.begin)
            return
        self._check_compound(lines, function.compound, indent)

    def check_source(self, source, source_tokens, pp_tokens, expr):
        lines = source.splitlines()
        compounds_to_check = list(expr.select('compound'))
        for struct in expr.select('struct'):
            if struct.compound is None:
                continue
            self._check_struct(lines, struct)
            compounds_to_check.remove(struct.compound)
        for func in expr.select('function_definition'):
            self._check_function_def(lines, func)
            compounds_to_check.remove(func.compound)
        for compound in compounds_to_check:
            self._check_compound(lines, compound)
        for child in expr.children:
            if not isinstance(child, DeclarationExpr):
                continue
            self._check_declaration(lines, child)


def get_argument_parser(checkers):
    descr = 'Check your C programs against the "EPITECH norm".'
    epilog = COPYRIGHT
    formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=descr,
                                     formatter_class=formatter,
                                     epilog=epilog)

    parser.add_argument('source_files',
                        nargs='*',
                        help='source files or directories to check')

    parser.add_argument('--include-dir', '-I',
                        action='append',
                        help="add a directory to the header search path")

    parser.add_argument('--verbose', '-v',
                        action='store_true',
                        help="verbose output")

    parser.add_argument('--warn', '-W',
                        action='store_true',
                        help="enable warnings")

    parser.add_argument('--json',
                        action='store_true',
                        help="JSON output (this is not really "
                        "human-readable)")

    help_str = 'tabulation width (defaults to {})'.format(
        StyleChecker.DEFAULT_TAB_WIDTH)
    parser.add_argument('--tab-width',
                        action='store',
                        default=StyleChecker.DEFAULT_TAB_WIDTH,
                        type=int,
                        help=help_str)

    for checker in checkers:
        checker.create_argument_group(parser)

    return parser


def colorize_string_csi(code, string, stop=True):
    return '\x1b[' + str(code) + 'm' + string + ('\x1b[0m' if stop else '')


def get_color_code(name):
    colors = 'black red green yellow blue magenta cyan white'.split()
    if name not in colors:
        raise Exception()
    return colors.index(name)


def colorize(style_name, string, bold=False):
    if bold:
        return colorize_string_csi(1, colorize(style_name, string),
                                   stop=False)
    color = get_color_code(style_name)
    return colorize_string_csi(color + 90, string)


def create_checkers(issue_handler):
    checkers_classes = [
        BinaryOpSpaceChecker,
        BraceChecker,
        CallChecker,
        CommaChecker,
        CommentChecker,
        DeclarationChecker,
        DeclaratorAlignmentChecker,
        DeclaratorChecker,
        DirectiveIndentationChecker,
        EmptyLineChecker,
        EmptyLineInFunctionChecker,
        FunctionCountChecker,
        FunctionLengthChecker,
        HeaderChecker,
        HeaderCommentChecker,
        IndentationChecker,
        KeywordSpaceChecker,
        LineLengthChecker,
        NameChecker,
        OneStatementByLineChecker,
        ParenChecker,
        ReturnChecker,
        SourceFileChecker,
        SupinfoChecker,
        TrailingWhitespaceChecker,
        UnaryOpSpaceChecker,
    ]
    return [c(issue_handler) for c in checkers_classes]


def _empty_file_issue(path, issue_handler):
    begin = Position(path)
    issue_handler(StyleIssue('Empty file, missing header comment', begin))


def check_open_file(open_file, checkers, include_dirs=None,
                    included_file_cache=None):
    """
    Check an open file against the norm

    Return a tuple (file_source, issues).
    """
    if include_dirs is None:
        include_dirs = []
    source = open_file.read()

    try:
        source_tokens = lex(source, open_file.name)
        if len(source_tokens) == 0:
            _empty_file_issue(open_file.name, checkers[0].issue)
            # We must return here since some checkers fails if there
            # is no token to check.
            return source, []

        pp_result = convert_to_pp_result(source_tokens,
                                         open_file.name,
                                         include_dirs)
        root_expr, issues = parse(pp_result,
                                  open_file.name,
                                  include_dirs, included_file_cache)

    except NSyntaxError as e:
        return source, [e]

    for checker in checkers:
        checker.check_source(source,
                             source_tokens,
                             pp_result.tokens,
                             root_expr)

    return source, issues


def check_file(file_path, checkers, include_dirs=None,
               included_file_cache=None):
    """
    Open a file and check it against the norm.

    Return a tuple (file_source, issues).
    """
    with open(file_path) as open_file:
        source, issues = check_open_file(open_file, checkers,
                                         include_dirs,
                                         included_file_cache)
    return source, issues


class Program:
    """
    The main program
    """

    def __init__(self):
        self.checkers = create_checkers(self._add_issue)

        argument_parser = get_argument_parser(self.checkers)
        options = argument_parser.parse_args()
        self.options = options

        for checker in self.checkers:
            checker.configure(options)

        self._issues = []
        self.include_dirs = options.include_dir
        if self.include_dirs is None:
            self.include_dirs = []
        self.verbose = options.verbose
        self.colors = os.isatty(sys.stdout.fileno())
        self.sources = {}
        self._included_file_cache = IncludedFileCache()

    def _add_issue(self, issue):
        assert isinstance(issue, StyleIssue)
        self._issues.append(issue)

    def _add_issues(self, issues):
        for issue in issues:
            self._add_issue(issue)

    def _get_line_at(self, position):
        """
        Returns the whole line at the given position or None
        """
        file_name = position.file_name
        if file_name not in self.sources:
            return None
        source = self.sources[file_name]
        lines = source.splitlines()
        if len(lines) == 0:
            return ''
        return lines[position.line - 1]

    def _get_visible_column(self, position):
        """
        Returns the visible column index at the given position or -1
        """
        line = self._get_line_at(position)
        if line is None:
            return -1
        left = line[:position.column - 1]
        # XXX: hack hack hack
        return self.checkers[0].get_visible_width(left, position)

    def print_issue(self, issue):
        if not self.options.warn and issue.level == 'warn':
            return

        bold = issue.level != 'note'
        string = self._colorize('white', str(issue.position) + ': ', bold)
        color = 'red'
        if issue.level == 'warn':
            color = 'yellow'
        if issue.level == 'note':
            color = 'blue'
        string += self._colorize(color, str(issue.level) + ': ', True)
        string += self._colorize('white', issue.message, bold)
        print(string)

        line = self._get_line_at(issue.position)
        if line is None:
            return
        print(line)
        visible_column_index = self._get_visible_column(issue.position)
        marker = self._colorize('green', '^', True)
        print(' ' * visible_column_index + marker)

    def _position_to_dict(self, position):
        return {
            'file_name': position.file_name,
            'index': position.index,
            'line': position.line,
            'column': position.column,
            'visibleColumn': self._get_visible_column(position),
        }

    def _issue_to_dict(self, issue):
        return {
            'level': issue.level,
            'message': issue.message,
            'position': self._position_to_dict(issue.position),
        }

    def _issues_to_dict(self):
        return [self._issue_to_dict(i) for i in self._issues]

    def print_issues(self):
        if self.options.json:
            import json
            print(json.dumps(self._issues_to_dict()))
            return
        for issue in self.issues:
            self.print_issue(issue)

    @property
    def issues(self):
        return self._issues[:]

    def check(self):
        if len(self.options.source_files) == 0:
            print('No input files')
            return
        for path in self.options.source_files:
            self.check_file_or_dir(path)

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

    def _colorize(self, style_name, string, bold=False):
        if not self.colors:
            return string
        return colorize(style_name, string, bold)

    def _check_dir(self, path, include_dirs):
        if not self.options.json:
            message = 'checking directory {!r}'.format(path)
            print(self._colorize('blue', message))
        include_dirs += get_include_dirs_from_makefile(path)
        for file_name in os.listdir(path):
            if file_name.startswith('.'):
                continue
            file_path = os.path.join(path, file_name)
            if os.path.isfile(file_path):
                if not (file_name.endswith('.c') or file_name.endswith('.h')):
                    continue
            self.check_file_or_dir(file_path, include_dirs)

    def _check_file(self, file_path, include_dirs):
        if self.verbose:
            print(self._colorize('black', file_path))
        source, issues = check_file(file_path,
                                    self.checkers,
                                    include_dirs,
                                    self._included_file_cache)
        assert source is not None
        self.sources[file_path] = source
        self._add_issues(issues)


def main():
    program = Program()
    program.check()
    program.print_issues()
    error_count = len([e for e in program.issues if e.level == 'error'])
    sys.exit(os.EX_OK if error_count == 0 else os.EX_DATAERR)


if __name__ == '__main__':
    main()
