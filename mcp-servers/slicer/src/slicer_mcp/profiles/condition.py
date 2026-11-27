"""Evaluate a vendor preset's compatibility condition against a config.

PrusaSlicer's vendor bundles say which print and filament presets suit a printer with
an expression such as::

    printer_notes=~/.*MK4.*/ and printer_notes!~/.*MK4IS.*/ and nozzle_diameter[0]==0.4

The bundles use a small part of the slicer's expression language: ``and``, ``or``,
``not`` (also written ``&&``, ``||``, ``!``), parentheses, ``==``, ``!=``, ``<``, ``<=``, ``>``, ``>=``, regular-expression
matches ``=~ /re/`` and ``!~ /re/``, variables with an optional ``[index]``, numbers,
quoted strings and bare boolean variables. This module evaluates exactly that part; an
expression outside it raises :class:`ConditionError`, so a preset is never matched by a
guess.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


class ConditionError(ValueError):
    """The expression uses syntax outside the supported subset."""


@dataclass(frozen=True)
class _Token:
    kind: str  # num, str, re, ident, op, lparen, rparen, lbrack, rbrack, word
    text: str


_TOKEN = re.compile(
    r"""\s*(?:
        (?P<num>-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)
      | "(?P<str>(?:[^"\\]|\\.)*)"
      | /(?P<re>(?:[^/\\]|\\.)*)/
      | (?P<op>=~|!~|==|!=|<=|>=|<|>|=)
      | (?P<andand>&&)
      | (?P<oror>\|\|)
      | (?P<bang>!)
      | (?P<lparen>\()
      | (?P<rparen>\))
      | (?P<lbrack>\[)
      | (?P<rbrack>\])
      | (?P<ident>[A-Za-z_][A-Za-z0-9_]*)
    )""",
    re.VERBOSE,
)


def _tokens(text: str) -> list[_Token]:
    out: list[_Token] = []
    pos = 0
    text = text.strip()
    while pos < len(text):
        match = _TOKEN.match(text, pos)
        if not match or match.end() == pos:
            raise ConditionError(f"cannot read {text[pos : pos + 20]!r} in {text!r}")
        pos = match.end()
        kind = match.lastgroup or ""
        value = match.group(kind)
        if kind == "ident" and value in {"and", "or", "not", "true", "false"}:
            kind = "word"
        if kind in {"bang", "andand", "oror"}:
            value = {"bang": "not", "andand": "and", "oror": "or"}[kind]
            kind = "word"
        out.append(_Token(kind, value))
    return out


def _scalar(value: Any, index: int | None) -> Any:
    """A config value as the expression sees it: list values are ``;``/``,`` separated."""
    if index is None:
        return value
    if isinstance(value, (list, tuple)):
        items = list(value)
    else:
        items = [p.strip() for p in re.split(r"[;,]", str(value))]
    if not items:
        raise ConditionError("index into an empty value")
    return items[min(index, len(items) - 1)]


def _number(value: Any) -> float | None:
    try:
        return float(str(value).strip().rstrip("%"))
    except ValueError:
        return None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"", "0", "false", "nil"}:
        return False
    number = _number(text)
    return number != 0 if number is not None else True


class _Parser:
    def __init__(self, tokens: list[_Token], config: Mapping[str, Any]) -> None:
        self.tokens, self.pos, self.config = tokens, 0, config

    def peek(self) -> _Token | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def take(self) -> _Token:
        token = self.peek()
        if token is None:
            raise ConditionError("the expression ends too early")
        self.pos += 1
        return token

    def parse(self) -> bool:
        value = self.or_expr()
        if self.peek() is not None:
            raise ConditionError(f"unexpected {self.peek().text!r}")  # type: ignore[union-attr]
        return value

    def or_expr(self) -> bool:
        value = self.and_expr()
        while (t := self.peek()) is not None and t.kind == "word" and t.text == "or":
            self.take()
            right = self.and_expr()
            value = value or right
        return value

    def and_expr(self) -> bool:
        value = self.not_expr()
        while (t := self.peek()) is not None and t.kind == "word" and t.text == "and":
            self.take()
            right = self.not_expr()
            value = value and right
        return value

    def not_expr(self) -> bool:
        token = self.peek()
        if token is not None and token.kind == "word" and token.text == "not":
            self.take()
            return not self.not_expr()
        return self.comparison()

    def comparison(self) -> bool:
        token = self.peek()
        if token is not None and token.kind == "lparen":
            self.take()
            value = self.or_expr()
            if self.take().kind != "rparen":
                raise ConditionError("a parenthesis is not closed")
            return value
        left = self.operand()
        op = self.peek()
        if op is None or op.kind != "op":
            return _truthy(left)
        self.take()
        if op.text in {"=~", "!~"}:
            pattern = self.take()
            if pattern.kind != "re":
                raise ConditionError(f"{op.text} needs /regex/")
            found = re.fullmatch(pattern.text, str(left), re.DOTALL) is not None
            return found if op.text == "=~" else not found
        right = self.operand()
        return _compare(left, op.text, right)

    def operand(self) -> Any:
        token = self.take()
        if token.kind == "num":
            return float(token.text)
        if token.kind == "str":
            return token.text.replace('\\"', '"')
        if token.kind == "word" and token.text in {"true", "false"}:
            return token.text == "true"
        if token.kind != "ident":
            raise ConditionError(f"unexpected {token.text!r}")
        if token.text not in self.config:
            raise ConditionError(f"unknown variable {token.text!r}")
        index = None
        nxt = self.peek()
        if nxt is not None and nxt.kind == "lbrack":
            self.take()
            number = self.take()
            if number.kind != "num" or self.take().kind != "rbrack":
                raise ConditionError("an index must be [number]")
            index = int(float(number.text))
        return _scalar(self.config[token.text], index)


def _compare(left: Any, op: str, right: Any) -> bool:
    a, b = _number(left), _number(right)
    if a is not None and b is not None:
        pair: tuple[Any, Any] = (a, b)
    else:
        pair = (str(left), str(right))
    x, y = pair
    if op in {"==", "="}:
        return bool(x == y)
    if op == "!=":
        return bool(x != y)
    if op == "<":
        return bool(x < y)
    if op == "<=":
        return bool(x <= y)
    if op == ">":
        return bool(x > y)
    if op == ">=":
        return bool(x >= y)
    raise ConditionError(f"unknown operator {op}")


def evaluate(expression: str, config: Mapping[str, Any]) -> bool:
    """True when ``config`` satisfies ``expression``; an empty expression is true.

    Raises:
        ConditionError: the expression uses syntax outside the supported subset or
            names a variable the config does not have.
    """
    text = (expression or "").strip()
    if not text:
        return True
    return _Parser(_tokens(text), config).parse()


__all__ = ["ConditionError", "evaluate"]
