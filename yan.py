#!/usr/bin/env python3

# pylint: disable=too-many-lines, redefined-variable-type

"""
A C brace style checker designed for the "EPITECH norm".
"""

import argparse
import os
import re
import sys


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


class AbstractIssue:
    """
    Represents a style issue reported by the checker.
    """

    def __init__(self, message, position):
        """
        message: A string describing the issue
        position: The Position of the issue
        """
        assert isinstance(message, str)
        assert isinstance(position, Position)
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

    def __str__(self):
        """
        Return an user-friendly string describing the issue and
        its position
        """
        return "{}: {}".format(self.position, self.message)


class NSyntaxError(Exception, AbstractIssue):
    """
    Represents a syntax error.
    """

    def __init__(self, message, position):
        Exception.__init__(self, message)
        AbstractIssue.__init__(self, message, position)

    def __str__(self):
        return AbstractIssue.__str__(self)


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


class IncludedFile:
    """
    Represents a file included with an #include directive.
    """

    def __init__(self, system, path):
        assert isinstance(system, bool)
        assert isinstance(path, str)
        self._system = system
        self._path = path

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
                self.__dict__ == other.__dict__)

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def system(self):
        return self._system

    @property
    def path(self):
        return self._path

    def __repr__(self):
        return '<IncludedFile system={!r} path={!r}>'.format(
            self.system, self.path)


class FileInclusion:
    """
    Represents an inclusion with an #include directive.
    """

    def __init__(self, token_index, file):
        """
        token_index: The index of the removed inclusion in the
        preprocessed token list.
        file: An IncludedFile
        """
        assert isinstance(file, IncludedFile)
        self._token_index = token_index
        self._file = file

    @property
    def token_index(self):
        return self._token_index

    @property
    def file(self):
        return self._file

    def __repr__(self):
        return '<FileInclusion file={!r} token_index={!r}>'.format(
            self.file, self.token_index)


class PreprocessorResult:
    def __init__(self, tokens, inclusions):
        self._tokens = tokens[:]
        self._inclusions = inclusions[:]

    @property
    def tokens(self):
        return self._tokens[:]

    @property
    def inclusions(self):
        return self._inclusions[:]

    def append(self, o):
        if isinstance(o, Token):
            self._tokens.append(o)
        elif isinstance(o, FileInclusion):
            self._inclusions.append(o)
        else:
            raise ValueError()

    def __add__(self, other):
        return PreprocessorResult(self.tokens + other.tokens,
                                  self.inclusions + other.inclusions)


def get_system_header_types():
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
        'grp.h':        '',
        'ncurses.h':    curses,
        'pwd.h':        '',
        'setjmp.h':     'jmp_buf',
        'signal.h':     'sighandler_t sigset_t',
        'stdarg.h':     'va_list',
        'stddef.h':     stddef,
        'stdint.h':     stdint,
        'stdio.h':      stddef + ' FILE',
        'stdlib.h':     stddef + ' div_t ldiv_t',
        'string.h':     stddef,
        'time.h':       stddef + ' clock_t time_t',
        'unistd.h':     stddef + ' ssize_t',
        'sys/stat.h':   '',
        'sys/types.h':  sys_types,
    }
    return {h: types.split() for h, types in strings.items()}


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


def preprocess_system_include(token_index, name):
    """
    Returns a FileInclusion on success.
    Returns None if there is no system header with the given name.
    """
    system_headers = get_system_header_types()
    if name not in system_headers:
        print('Warning: Header not found: {!r}'.format(name))
        return None
    included_file = IncludedFile(True, name)
    return FileInclusion(token_index, included_file)


def preprocess_local_include(token_index, path):
    """
    Returns a FileInclusion on success.
    """
    assert os.path.exists(path)
    included_file = IncludedFile(False, path)
    return FileInclusion(token_index, included_file)


def preprocess_include(token_index, directive_string, position, include_dirs):

    def get_name_and_system(directive_string, position):
        """
        Return a tuple (name, system) where 'name' is the included file
        name specified between the quotes, and 'system' is True if
        '#include <...>' has been used, False if '#include "..."' has
        been used.
        """
        assert directive_string.startswith('include')
        quoted_name = directive_string[len('include'):].strip()
        name = quoted_name[1:-1]
        if quoted_name.startswith('<') and quoted_name.endswith('>'):
            return name, True
        elif quoted_name.startswith('"') and quoted_name.endswith('"'):
            return name, False
        else:
            raise SyntaxError('Invalid #include directive', position)

    name, system = get_name_and_system(directive_string, position)
    dir_path = os.path.dirname(position.file_name)
    included_file_path = get_included_file_path(dir_path,
                                                name, system,
                                                include_dirs)
    if included_file_path is None:
        return preprocess_system_include(token_index, name)
    else:
        return preprocess_local_include(token_index, included_file_path)


def preprocess_directive(token_index, directive, include_dirs):
    """
    Returns a FileInclusion or None.
    """
    assert directive.kind == 'directive'
    string = directive.string.strip()
    assert string[0] == '#'
    # There may be several spaces between the '#' and the
    # next word, we need to strip these
    string = string[1:].strip()
    if string.startswith('include'):
        return preprocess_include(token_index, string, directive.begin,
                                  include_dirs)
    if string.startswith('ifdef'):
        string = string[5].strip()
        return None


def preprocess(tokens, include_dirs=None):
    """
    Warning: This is not a real C preprocessor.

    This function acts a bit like a preprocessor since it removes
    comments and expands includes.
    """
    if include_dirs is None:
        include_dirs = []

    result = PreprocessorResult([], [])
    for token in tokens:
        if token.kind == 'comment':
            continue
        elif token.kind == 'directive':
            i = len(result.tokens)
            fi = preprocess_directive(i, token, include_dirs)
            if fi is not None:
                result.append(fi)
        else:
            result.append(token)
    return result


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

    def __init__(self, tokens, include_dirs=None):
        if include_dirs is None:
            include_dirs = []
        self._source_tokens = tokens[:]
        self._include_dirs = include_dirs[:]
        pp_result = preprocess(tokens, include_dirs)
        self._inclusions = pp_result.inclusions[:]

        TokenReader.__init__(self, pp_result.tokens)
        self._types = []
        self._included_files = []
        self._in_typedef = False

    @property
    def file_name(self):
        """
        Return the file name or '<unknown file>'.
        """
        tokens = self._source_tokens
        if len(tokens) == 0:
            return '<unknown file>'
        return tokens[0].begin.file_name

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
        assert not included_file.system

        if self._is_already_included(included_file):
            return
        self._included_files.append(included_file)

        path = included_file.path
        assert os.path.exists(path)
        with open(path) as h:
            source = h.read()
        tokens = lex(source, file_name=path)
        parser = Parser(tokens, self._include_dirs)
        parser.add_types(self.types)
        parser._included_files += self._included_files
        try:
            parser.parse()
        except NSyntaxError as e:
            print("In file included from {}:".format(self.file_name))
            raise e
        self.add_types(parser.types)

    def _expand_system_include(self, included_file):
        assert included_file.system

        header_types = get_system_header_types()
        name = included_file.path
        assert name in header_types

        if self._is_already_included(included_file):
            return

        self._included_files.append(included_file)
        self.add_types(header_types[name])

    def _expand_included_file(self, included_file):
        assert isinstance(included_file, IncludedFile)
        # print('expand included file {!r}'.format(included_file.path))
        if included_file.system:
            self._expand_system_include(included_file)
        else:
            self._expand_local_include(included_file)

    def _expand_inclusions(self):
        while len(self._inclusions) > 0:
            inclusion = self._inclusions[0]
            if inclusion.token_index != self.index:
                break
            self._inclusions.pop(0)
            self._expand_included_file(inclusion.file)

    def next(self):
        self._expand_inclusions()
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
            left = BinaryOperationExpr(left, operator, sub_function())
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
        self._expand_inclusions()
        if len(self._inclusions) > 0:
            raise Exception()
        return expr

    def parse(self):
        return self._parse_translation_unit()


def argument_to_tokens(v):
    if isinstance(v, str):
        v = lex(v)
    if not isinstance(v, list):
        raise ValueError('Expected a list of tokens')
    return v


def parse(v, include_directories=None):
    """
    v: a string or a list of tokens
    """

    if include_directories is None:
        include_directories = []

    tokens = argument_to_tokens(v)
    parser = Parser(tokens, include_directories)
    tree = parser.parse()
    assert isinstance(tree, Expr)
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


class StyleIssue(AbstractIssue):
    def __init__(self, message, position, level='error'):
        assert level in 'warn error'.split()
        super().__init__(message, position)
        self._level = level

    @property
    def level(self):
        return self._level


class StyleChecker:
    def __init__(self, issue_handler, options):
        self._issue_handler = issue_handler
        self._tab_width = options.tab_width
        self._indent_chars = ' \t'

    @property
    def tab_width(self):
        return self._tab_width

    @property
    def indent_chars(self):
        return self._indent_chars

    def issue(self, issue):
        assert isinstance(issue, StyleIssue)
        self._issue_handler(issue)

    def warn(self, message, position):
        self.issue(StyleIssue(message, position, level='warn'))

    def error(self, message, position):
        self.issue(StyleIssue(message, position))

    def check(self, tokens, expr):
        raise NotImplementedError()

    def check_source(self, source, tokens, expr):
        """
        Like a StyleChecker, but accepts a string containing the source
        code of the file
        """
        # pylint: disable=unused-argument
        return self.check(tokens, expr)

    def check_same_line(self, token_a, token_b):
        """
        Check if the two given tokens are on the same line
        """
        assert isinstance(token_a, Token)
        assert isinstance(token_b, Token)

        if token_a.end.line != token_b.start.line:
            msg = '{!r} is not on the same line than {!r}'.format(
                token_a.string, token_b.string)
            self.error(msg, token_a.end)

    @staticmethod
    def _pop_char(string):
        if len(string) == 0:
            raise Exception()
        return string[0], string[1:]

    def get_visible_width(self, indent_string, position, column=-1):
        """
        Returns the visible width in the given string.

        string: A string constitued of spaces and tabs.
        """
        if column == -1:
            column = position.column
        begin = column
        tabs = 0
        spaces = 0
        while len(indent_string) > 0:
            char, indent_string = StyleChecker._pop_char(indent_string)
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
        return column - begin

    def check_indent(self, string, expected_width, position, column=-1):
        indent_string = ''
        while string[0] in self.indent_chars:
            indent_string += string[0]
            string = string[1:]

        if column == -1:
            column = position.column
        width = self.get_visible_width(indent_string, position, column)
        if width != expected_width:
            self.error('Bad indent level, expected {} space(s), got {}'.format(
                expected_width, width), position)

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
    def __init__(self, issue_handler, options):
        super().__init__(issue_handler, options)

    def check_line(self, begin, line, end):
        """
        This function is called on each line of the file.

        begin: The position of the first character of the line.
        line: The string of the line.
        end: The position of the last character of the line.
        """
        raise NotImplementedError()

    def check_source(self, source, tokens, expr):
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
        super().__init__(issue_handler, options)

    def check_line(self, begin, line, end):
        if len(line) > 80:
            self.error("Line too long (more than 80 characters)", end)


class TrailingWhitespaceChecker(LineChecker):
    def __init__(self, issue_handler, options):
        super().__init__(issue_handler, options)

    def check_line(self, begin, line, end):
        if line.rstrip() != line:
            self.warn("Trailing whitespaces at the end of the line", end)


class HeaderCommentChecker(StyleChecker):
    def __init__(self, issue_handler, options):
        self.username_check_enabled = options.header_username
        super().__init__(issue_handler, options)

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
        super().__init__(issue_handler, options)

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
        for token in tokens:
            if token.kind == 'comment':
                self.check_comment(token)

        funcs = expr.select('function_definition')
        for func in funcs:
            func_tokens = self.get_all_tokens_in_expr(tokens, func)
            for token in func_tokens:
                if token.kind == 'comment':
                    self.error('Comment inside a function', token.begin)


class BinaryOpSpaceChecker(StyleChecker):
    def __init__(self, issue_handler, options):
        super().__init__(issue_handler, options)

    def check_source(self, source, tokens, expr):
        bin_ops = expr.select('binary_operation')
        for operation in bin_ops:
            left_token = operation.left.last_token
            right_token = operation.right.first_token
            operator = operation.operator
            margin = 0 if operator.string in '. ->'.split() else ' '
            self.check_margin(source, left_token, margin, operator)
            self.check_margin(source, operator, margin, right_token)


class UnaryOpSpaceChecker(StyleChecker):
    def __init__(self, issue_handler, options):
        super().__init__(issue_handler, options)

    def check_source(self, source, tokens, expr):
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
    def __init__(self, issue_handler, options):
        super().__init__(issue_handler, options)

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
        super().__init__(issue_handler, options)

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
        super().__init__(issue_handler, options)

    def check(self, tokens, expr):
        funcs = expr.select('function_definition')
        if len(funcs) > 5:
            self.error('Too many functions (more than 5)',
                       expr.first_token.begin)


class DirectiveIndentationChecker(StyleChecker):
    def __init__(self, issue_handler, options):
        super().__init__(issue_handler, options)

    def check(self, tokens, expr):
        level = 0
        for token in tokens:
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
    def __init__(self, issue_handler, options):
        super().__init__(issue_handler, options)

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


class IndentationChecker(StyleChecker):
    def __init__(self, issue_handler, options):
        super().__init__(issue_handler, options)
        self.level = 0

    def check_begin_indentation(self, lines, expr):
        if isinstance(expr, Expr):
            token = expr.tokens[0]
        else:
            token = expr
        begin = token.begin
        first_line = lines[begin.line - 1]
        self.check_indent(first_line, self.level, begin, 1)

    def check_end_indentation(self, lines, expr):
        end = expr.tokens[-1].end
        first_line = lines[end.line - 1]
        self.check_indent(first_line, self.level, end, 1)

    def check_expr(self, lines, expr):
        indented_classes = (
            FunctionDefinitionExpr,
            CompoundExpr,
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
            self.check_begin_indentation(lines, expr)

        if isinstance(expr, indentor_classes):
            self.level += 2

        if isinstance(expr, IfExpr):
            self.check_expr(lines, expr.expression)
            self.check_expr(lines, expr.statement)
            if expr.else_statement is not None:
                self.level -= 2
                self.check_begin_indentation(lines, expr.else_token)
                self.level += 2
                if isinstance(expr.else_statement, IfExpr):
                    self.level -= 2
                self.check_expr(lines, expr.else_statement)
                if isinstance(expr.else_statement, IfExpr):
                    self.level += 2
        else:
            for child in expr.children:
                self.check_expr(lines, child)

        if isinstance(expr, indentor_classes):
            self.level -= 2

        if isinstance(expr, CompoundExpr):
            self.check_end_indentation(lines, expr)

    def check_source(self, source, tokens, expr):
        self.level = 0
        self.check_expr(source.splitlines(), expr)
        if self.level != 0:
            self.warn('The indentation seems inconsistent', tokens[0].begin)


def get_argument_parser():
    descr = 'Check your C programs against the "EPITECH norm".'
    parser = argparse.ArgumentParser(description=descr)

    parser.add_argument('source_files',
                        nargs='*',
                        help='source files or directories to check')

    parser.add_argument('-I',
                        action='append',
                        help="add a directory to the header search path")

    parser.add_argument('--verbose', '-v',
                        action='store_true',
                        help="verbose output")

    parser.add_argument('--header-username',
                        action='store_true',
                        help="check the username in header comments")

    parser.add_argument('--tab-width',
                        action='store',
                        default=8,
                        type=int,
                        help="tabulation width (defaults to 8)")

    return parser


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
            IndentationChecker,
            LineLengthChecker,
            ReturnChecker,
            TrailingWhitespaceChecker,
            UnaryOpSpaceChecker,
        ]
        self.verbose = args.verbose
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
        self._print_fg_color('blue', 'checking directory {!r}'.format(path))
        include_dirs += get_include_dirs_from_makefile(path)
        for file_name in os.listdir(path):
            if file_name.startswith('.'):
                continue
            file_path = os.path.join(path, file_name)
            if os.path.isfile(file_path):
                if not (file_name.endswith('.c') or file_name.endswith('.h')):
                    continue
            self.check_file_or_dir(file_path, include_dirs)

    def _print_fg_color(self, color_name, string):
        if self.colors:
            print_fg_color(color_name, string)
        else:
            print(string)

    def _empty_file_error(self, path):
        begin = Position(path)
        print_issue(StyleIssue('Empty file, missing header comment', begin))

    def _check_file(self, path, include_dirs):
        if self.verbose:
            self._print_fg_color('black', path)
        with open(path) as source_file:
            source = source_file.read()
            tokens = lex(source, file_name=path)
            if len(tokens) == 0:
                self._empty_file_error(path)
                # We must return here since some checkers fails if there
                # is no token to check.
                return

            root_expr = parse(tokens, include_dirs)
            for checker in self.checkers:
                checker.check_source(source, tokens, root_expr)


def main():
    argument_parser = get_argument_parser()
    args = argument_parser.parse_args()

    program = Program(args)
    for source_file in args.source_files:
        program.check_file_or_dir(source_file)


if __name__ == '__main__':
    main()
