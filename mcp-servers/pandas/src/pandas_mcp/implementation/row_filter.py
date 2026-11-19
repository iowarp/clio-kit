"""Filter rows with a query-style condition.

Tools accept conditions such as ``"revenue < 50000 and region == 'West'"``.
``DataFrame.query`` evaluates the whole string and resolves ``@name`` in the
calling frame, which is more than these tools need. This module parses the
condition with ``ast`` and builds the mask from a fixed set of operations.

Supported:
- column names, bare or in backticks (`` `unit price` ``);
- constants (strings, numbers, booleans, None) and lists or tuples of them;
- comparisons ``== != < <= > >=``, chained (``0 < x <= 5``), and
  ``in`` / ``not in`` against a list;
- ``and``, ``or``, ``not``, and ``&``, ``|``, ``~``;
- ``+ - * / // %`` where at least one side is a column.

Everything else, including ``@`` references, attribute access, calls and
names that are not columns, is refused with a ValueError.
"""

import ast
import operator
import re
from typing import Any, Callable, Dict

import pandas as pd

MAX_CONDITION_LENGTH = 4000

_COMPARE: Dict[type, Callable[[Any, Any], Any]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}

_ARITHMETIC: Dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}

_BACKTICK = re.compile(r"`([^`]*)`")


class _Evaluator:
    """Evaluate one parsed condition against a DataFrame."""

    def __init__(self, df: pd.DataFrame, aliases: Dict[str, str]):
        self.df = df
        self.aliases = aliases

    def value(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            if node.value is None or isinstance(node.value, (str, int, float, bool)):
                return node.value
            raise ValueError("Unsupported constant in filter condition")
        if isinstance(node, ast.Name):
            name = self.aliases.get(node.id, node.id)
            if name in self.df.columns:
                return self.df[name]
            if node.id in ("True", "False", "None"):
                return {"True": True, "False": False, "None": None}[node.id]
            raise ValueError(f"Unknown column in filter condition: {name!r}")
        if isinstance(node, (ast.List, ast.Tuple)):
            items = [self.value(item) for item in node.elts]
            if any(isinstance(item, pd.Series) for item in items):
                raise ValueError("Lists in a filter condition must hold constants")
            return items
        if isinstance(node, ast.UnaryOp):
            operand = self.value(node.operand)
            if isinstance(node.op, (ast.Not, ast.Invert)):
                return ~self._mask(operand)
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.UAdd):
                return +operand
        if isinstance(node, ast.BoolOp):
            masks = [self._mask(self.value(v)) for v in node.values]
            result = masks[0]
            for mask in masks[1:]:
                result = (
                    result & mask if isinstance(node.op, ast.And) else result | mask
                )
            return result
        if isinstance(node, ast.BinOp):
            left, right = self.value(node.left), self.value(node.right)
            if isinstance(node.op, (ast.BitAnd, ast.BitOr)):
                left, right = self._mask(left), self._mask(right)
                return left & right if isinstance(node.op, ast.BitAnd) else left | right
            op = _ARITHMETIC.get(type(node.op))
            if op is None:
                raise ValueError("Unsupported operator in filter condition")
            # Arithmetic on two constants is never needed to filter rows, and
            # would let a condition like 'x' * 10**12 exhaust memory.
            if not (isinstance(left, pd.Series) or isinstance(right, pd.Series)):
                raise ValueError("Arithmetic in a filter condition needs a column")
            return op(left, right)
        if isinstance(node, ast.Compare):
            return self._compare(node)
        raise ValueError(
            f"Unsupported expression in filter condition: {type(node).__name__}"
        )

    def _compare(self, node: ast.Compare) -> pd.Series:
        left = self.value(node.left)
        result = None
        for op, comparator in zip(node.ops, node.comparators):
            right = self.value(comparator)
            if isinstance(op, (ast.In, ast.NotIn)):
                if not isinstance(left, pd.Series):
                    raise ValueError("'in' needs a column on its left")
                values = right if isinstance(right, list) else [right]
                part = left.isin(values)
                if isinstance(op, ast.NotIn):
                    part = ~part
            else:
                compare = _COMPARE.get(type(op))
                if compare is None:
                    raise ValueError("Unsupported comparison in filter condition")
                part = compare(left, right)
            mask = self._mask(part)
            result = mask if result is None else result & mask
            left = right
        return result

    def _mask(self, value: Any) -> pd.Series:
        if isinstance(value, pd.Series) and pd.api.types.is_bool_dtype(value):
            # Nullable columns compare to the "boolean" dtype; missing is false.
            return value.fillna(False).astype(bool)
        if isinstance(value, bool):
            return pd.Series(value, index=self.df.index)
        raise ValueError("Filter condition must evaluate to true or false per row")


def filter_rows(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    """Return the rows of ``df`` for which ``condition`` holds.

    Raises:
        ValueError: if the condition uses anything outside the supported subset
            (see the module docstring) or does not yield one boolean per row.
    """
    if len(condition) > MAX_CONDITION_LENGTH:
        raise ValueError("Filter condition is too long")
    aliases: Dict[str, str] = {}

    def alias(match: "re.Match[str]") -> str:
        name = f"__column_{len(aliases)}__"
        aliases[name] = match.group(1)
        return name

    text = _BACKTICK.sub(alias, condition)
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid filter condition: {exc.msg}") from exc
    evaluator = _Evaluator(df, aliases)
    return df[evaluator._mask(evaluator.value(tree.body))]
