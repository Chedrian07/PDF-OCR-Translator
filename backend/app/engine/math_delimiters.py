r"""Conservative presentation recovery for OCR's delimiter-free algebra.

This is deliberately a small language, not a general math detector: a relation
with spaces on both sides, short symbolic operands, and a numeric super/subscript
must all be present. Unsupported or ambiguous notation stays literal. Source
characters are retained; inline spans gain ``\(...\)`` wrappers. A separate
whole-line grammar also accepts explicit mathematical function calls and Greek
names, retaining their spelling with safe TeX presentation commands. The same
bounded, line-oriented stream is used for saved and live output.
"""

from __future__ import annotations

import re

_MAX_LINE = 4096
_MAX_FORMULA = 256
_WORD = re.compile(r"[A-Za-z]+")
_NUMBER = re.compile(r"\d+(?:\.\d+)?", re.ASCII)
_SCRIPT = re.compile(r"(?:\{[+-]?\d{1,3}\}|\d)", re.ASCII)
_SUBSCRIPT = re.compile(r"(?:\{(?:[+-]?\d{1,3}|[A-Za-z])\}|[A-Za-z0-9])", re.ASCII)
_GREEK_NAMES = frozenset(
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
    "omicron pi rho sigma tau upsilon phi chi psi omega".split()
)
_MATH_FUNCTIONS = frozenset("sqrt sum sin cos tan log ln exp abs".split())
_FENCE = re.compile(r" {0,3}(`{3,}|~{3,})(.*)")
_URL = re.compile(r"(?:[A-Za-z][A-Za-z0-9+.-]*://|www\.|mailto:)\S+")
_HTML_CODE = re.compile(r"<(pre|code|script|style|textarea)\b", re.IGNORECASE)
_TAG_START = re.compile(r"</?[A-Za-z]|<\|")
_GROUNDING_CODE = re.compile(
    r"<\|ref\|>(?:code|algorithm)<\|/ref\|><\|det\|>[^<]*<\|/det\|>"
    r"|<\|det\|>\s*(?:code|algorithm)\s*\[[^<]*<\|/det\|>",
    re.IGNORECASE,
)
_GROUNDING_START = re.compile(r"<\|ref\|>|<\|det\|>|<PAGE>")
_GROUNDING_PREFIX = re.compile(
    r"(?:(?:<\|ref\|>[^<]*<\|/ref\|>)?<\|det\|>[^<]*<\|/det\|>)*"
)
_CODE_LINE = re.compile(
    r"(?:^|<\|/det\|>)\s*(?:const|let|var|def|class|return|import|from|export|using"
    r"|if|elif|else|while|for|function|int|float|double|bool|auto|val|fun)\b"
)
_INDENTED_CODE = re.compile(r"(?:^|<\|/det\|>)(?: {4}|\t)")


class _Algebra:
    """Bounded recursive descent; no evaluation or source-to-TeX conversion."""

    def __init__(self, text: str, start: int):
        self.text = text
        self.pos = start
        self.limit = min(len(text), start + _MAX_FORMULA)
        self.symbols = 0
        self.script = False
        self.symbol_width = 1

    def spaces(self):
        while self.pos < self.limit and self.text[self.pos] in " \t":
            self.pos += 1

    def atom(self, depth=0):
        self.spaces()
        if self.pos >= self.limit or depth > 4:
            return False
        if self.text[self.pos] in "+-":
            self.pos += 1
            self.spaces()
        if self.pos >= self.limit:
            return False
        if self.text[self.pos] == "(":
            self.pos += 1
            if not self.expression(depth + 1):
                return False
            self.spaces()
            if self.pos >= self.limit or self.text[self.pos] != ")":
                return False
            self.pos += 1
        else:
            match = _WORD.match(self.text, self.pos, self.limit)
            if match:
                if not self.word(match, depth):
                    return False
            else:
                match = _NUMBER.match(self.text, self.pos, self.limit)
                if not match:
                    return False
                self.pos = match.end()
        scripts = set()
        while self.pos < self.limit and self.text[self.pos] in "^_":
            marker = self.text[self.pos]
            if marker in scripts:
                return False
            scripts.add(marker)
            match = self.script_match(marker)
            if not match:
                return False
            self.script = True
            self.pos = match.end()
        return True

    def word(self, match, depth):
        if len(match[0]) > self.symbol_width:
            return False
        self.symbols += len(match[0])
        self.pos = match.end()
        return True

    def script_match(self, marker):
        return _SCRIPT.match(self.text, self.pos + 1, self.limit)

    def product(self, depth):
        if not self.atom(depth):
            return False
        while True:
            end = self.pos
            self.spaces()
            if self.pos >= self.limit or self.text[self.pos] not in "*/":
                self.pos = end
                return True
            self.pos += 1
            if not self.atom(depth):
                return False

    def expression(self, depth=0):
        if not self.product(depth):
            return False
        while True:
            end = self.pos
            self.spaces()
            if self.pos >= self.limit or self.text[self.pos] not in "+-":
                self.pos = end
                return True
            self.pos += 1
            if not self.product(depth):
                return False

    def formula_end(self):
        if not self.expression() or not self.symbols:
            return None
        before = self.pos
        self.spaces()
        # A spaced relation excludes assignments in queries, paths, and most
        # configuration syntax. Multi-letter identifiers on the left are prose
        # or code more often than algebra, so they are intentionally unsupported.
        if before == self.pos or self.pos >= self.limit:
            return None
        relation = re.match(
            r"(?:<=|>=|=|<|>|≤|≥|≈)", self.text[self.pos : self.pos + 2]
        )
        if not relation:
            return None
        self.pos += len(relation[0])
        before = self.pos
        self.spaces()
        if before == self.pos:
            return None
        self.symbol_width = 2  # compact products such as xy or mv, never words
        if not self.expression() or not self.script:
            return None
        end = self.pos
        if end == self.limit and end < len(self.text):
            return None
        if end < len(self.text):
            next_char = self.text[end]
            if next_char.isalnum() or next_char in "_\\/@?#%&=^+*({<>":
                return None
            if (
                next_char == "."
                and end + 1 < len(self.text)
                and self.text[end + 1].isalnum()
            ):
                return None
            # Do not recover a fragment of a relation chain or unsupported
            # compound operator. The scanner also rejects starts after these.
            after = end
            while after < len(self.text) and self.text[after] in " \t":
                after += 1
            if self.text.startswith(("=", "<", ">", "≤", "≥", "≈"), after):
                return None
        return end


class _NamedAlgebra(_Algebra):
    """Whole-line extension; unknown identifiers/functions are never accepted."""

    def __init__(self, text):
        super().__init__(text, 0)
        self.function = False
        self.numeric_power = False

    def word(self, match, depth):
        name = match[0]
        self.pos = match.end()
        if len(name) == 1 or name.lower() in _GREEK_NAMES:
            self.symbols += 1
            return True
        if name not in _MATH_FUNCTIONS:
            return False
        self.spaces()
        if self.pos >= self.limit or self.text[self.pos] != "(":
            return False
        self.pos += 1
        if not self.expression(depth + 1):
            return False
        self.spaces()
        if self.pos >= self.limit or self.text[self.pos] != ")":
            return False
        self.pos += 1
        self.function = True
        return True

    def script_match(self, marker):
        pattern = _SUBSCRIPT if marker == "_" else _SCRIPT
        match = pattern.match(self.text, self.pos + 1, self.limit)
        if match and marker == "^":
            self.numeric_power = True
        return match


def _whole_line_math(line: str, prefix: int) -> str | None:
    body = line[prefix:].strip(" \t\r\n")
    if len(body) > _MAX_FORMULA:
        return None
    parser = _NamedAlgebra(body)
    if (
        parser.formula_end() != len(body)
        or not parser.function
        or not parser.numeric_power
    ):
        return None

    def present_word(match):
        name = match[0]
        if name in _MATH_FUNCTIONS:
            return r"\operatorname{" + name + "}"
        if len(name) > 1:
            return r"\mathrm{" + name + "}"
        return name

    start = prefix + len(line[prefix:]) - len(line[prefix:].lstrip(" \t\r\n"))
    return (
        line[:start]
        + r"\("
        + _WORD.sub(present_word, body)
        + r"\)"
        + line[start + len(body) :]
    )


class MathDelimiterStream:
    """Recover complete lines while retaining Markdown literal-region state.

    At most 4096 characters are buffered. A longer line disables inference for
    the rest of that page, because literal context beyond the lookahead is
    uncertain. OCR page markers reset state, as in the app's page renderer.
    """

    def __init__(self, sink):
        self.sink = sink
        self.pending = ""
        self.page_pending = ""
        self.long_line = False
        self.fence = None
        self.literal_end = None
        self.grounding_code = False
        self.in_tag = False
        self.tag_quote = None
        self.disabled = False

    def on_text(self, text: str):
        # Frame pages before line buffering so layout's page-at-a-time parsing
        # has exactly the same literal context as live and committed Markdown.
        parts = (self.page_pending + text).split("<PAGE>")
        for part in parts[:-1]:
            self._on_page_text(part)
            self._end_page()
            self.sink.on_text("<PAGE>")
            self.fence = self.literal_end = self.tag_quote = None
            self.grounding_code = self.in_tag = self.disabled = False
            self.long_line = False
        tail = parts[-1]
        keep = 0
        for size in range(1, len("<PAGE>")):
            if tail.endswith("<PAGE>"[:size]):
                keep = size
        self.page_pending = tail[-keep:] if keep else ""
        self._on_page_text(tail[:-keep] if keep else tail)

    def _on_page_text(self, text: str):
        pieces = text.splitlines(keepends=True)
        for piece in pieces:
            # Only LF closes a line: splitlines also recognizes characters that
            # Markdown treats as ordinary text, so those pieces simply accumulate.
            complete = piece.endswith("\n")
            if (
                self.long_line
                or len(self.pending) + len(piece.rstrip("\n")) > _MAX_LINE
            ):
                self.disabled = True
                self.sink.on_text(self.pending + piece)
                self.pending = ""
                self.long_line = not complete
            else:
                self.pending += piece
                if complete:
                    self.sink.on_text(self._line(self.pending))
                    self.pending = ""

    def end(self):
        self._on_page_text(self.page_pending)
        self.page_pending = ""
        self._end_page()

    def _end_page(self):
        if self.pending:
            self.sink.on_text(self._line(self.pending))
            self.pending = ""

    def _line(self, line: str) -> str:
        if self.disabled:
            return line
        # Grounding headers occupy no Markdown columns after postprocessing.
        prefix = _GROUNDING_PREFIX.match(line).end()
        fence = _FENCE.match(line, prefix)
        if self.fence:
            char, width = self.fence
            if (
                fence
                and fence[1][0] == char
                and len(fence[1]) >= width
                and not fence[2].strip()
            ):
                self.fence = None
            return line
        if not self.literal_end and fence:
            self.fence = (fence[1][0], len(fence[1]))
            return line
        if not self.literal_end and (
            _INDENTED_CODE.search(line) or _CODE_LINE.search(line)
        ):
            return line
        if not (
            self.literal_end
            or self.in_tag
            or self.grounding_code
            or _GROUNDING_CODE.match(line)
        ):
            whole_line = _whole_line_math(line, prefix)
            if whole_line is not None:
                return whole_line

        out, copied, pos = [], 0, 0
        while pos < len(line):
            if self.in_tag:
                pos = self._tag_end(line, pos)
                continue
            if self.grounding_code:
                next_block = _GROUNDING_START.search(line, pos)
                if not next_block:
                    break
                pos = next_block.start()
                self.grounding_code = False
            if self.literal_end:
                closer = self.literal_end
                if closer.startswith("</"):
                    match = re.search(
                        re.escape(closer[:-1]) + r"\s*>", line[pos:], re.IGNORECASE
                    )
                    if not match:
                        break
                    pos += match.end()
                else:
                    end = line.find(closer, pos)
                    if closer.startswith("`"):
                        while end >= 0 and (
                            (end > 0 and line[end - 1] == "`")
                            or (
                                end + len(closer) < len(line)
                                and line[end + len(closer)] == "`"
                            )
                        ):
                            end = line.find(closer, end + len(closer))
                    if end < 0:
                        break
                    pos = end + len(closer)
                self.literal_end = None
                continue
            char = line[pos]
            code_block = _GROUNDING_CODE.match(line, pos) if char == "<" else None
            if code_block:
                self.grounding_code = True
                pos = code_block.end()
                continue
            if line.startswith("<!--", pos):
                self.literal_end = "-->"
                pos += 4
                continue
            html = _HTML_CODE.match(line, pos)
            if html:
                # HTML code regions may span lines. Case-insensitive closing tags
                # are handled by protecting the whole line when needed below.
                closer = re.search(
                    rf"</{html[1]}\s*>", line[html.end() :], re.IGNORECASE
                )
                if closer:
                    pos = html.end() + closer.end()
                else:
                    self.literal_end = f"</{html[1].lower()}>"
                    pos = len(line)
                continue
            if char == "<" and _TAG_START.match(line, pos):
                self.in_tag = True
                pos = self._tag_end(line, pos + 1)
                continue
            if char == "`":
                end = pos + 1
                while end < len(line) and line[end] == "`":
                    end += 1
                if end - pos >= 3 and line.find(line[pos:end], end) < 0:
                    # A fence inside a quote/list is outside this small parser's
                    # indentation grammar. Protect it conservatively until a
                    # standalone closer or the next OCR page.
                    self.fence = ("`", end - pos)
                    break
                self.literal_end = line[pos:end]
                pos = end
                continue
            triple = next((x for x in ('"""', "'''") if line.startswith(x, pos)), None)
            if triple:
                self.literal_end = triple
                pos += len(triple)
                continue
            opening = next(
                (x for x in (r"\(", r"\[", "$$") if line.startswith(x, pos)), None
            )
            if opening:
                self.literal_end = {r"\(": r"\)", r"\[": r"\]", "$$": "$$"}[opening]
                pos += len(opening)
                continue
            if char in "$\"'":
                end = line.find(char, pos + 1)
                if end >= 0:
                    pos = end + 1
                    continue
                if char == "$":
                    break  # currency or a malformed delimiter: keep it literal
            url_start = char.isalpha() and (
                not pos or not (line[pos - 1].isalnum() or line[pos - 1] in ".+-")
            )
            url = _URL.match(line, pos) if url_start else None
            if url:
                pos = url.end()
                continue
            if char == "[":
                end = line.find("]", pos + 1)
                if end >= 0:
                    pos = end + 1
                    # Protect link destinations too; a bracketed expression may
                    # be a reference link even without an explicit destination.
                    if pos < len(line) and line[pos] == "(":
                        depth = 1
                        pos += 1
                        while pos < len(line) and depth:
                            depth += (line[pos] == "(") - (line[pos] == ")")
                            pos += 1
                    continue
            if char == "\\":
                if line.startswith("\\`", pos):
                    break  # an escaped code marker is literal documentation
                pos += 2
                continue
            if (char.isascii() and char.isalpha()) or char == "(":
                previous = line[pos - 1] if pos else ""
                if previous and (previous.isalnum() or previous in "_\\/@.:?#%&=^-"):
                    pos += 1
                    continue
                previous_index = pos - 1
                while previous_index >= 0 and line[previous_index] in " \t":
                    previous_index -= 1
                if previous_index >= 0 and line[previous_index] in "=+-*/^_≤≥≈":
                    pos += 1
                    continue
                end = _Algebra(line, pos).formula_end()
                if end is not None:
                    out.extend((line[copied:pos], r"\(", line[pos:end], r"\)"))
                    copied = pos = end
                    continue
            pos += 1
        out.append(line[copied:])
        return "".join(out)

    def _tag_end(self, line: str, pos: int) -> int:
        """Skip attributes without mistaking a quoted '>' for the tag end."""
        while pos < len(line):
            char = line[pos]
            pos += 1
            if self.tag_quote:
                if char == self.tag_quote:
                    self.tag_quote = None
            elif char in "\"'":
                self.tag_quote = char
            elif char == ">":
                self.in_tag = False
                break
        return pos


def recover_math_delimiters(text: str) -> str:
    """Add wrappers only; retain code, existing math, and rejected source exactly."""

    class Sink:
        def __init__(self):
            self.pieces = []

        def on_text(self, piece):
            self.pieces.append(piece)

    sink = Sink()
    stream = MathDelimiterStream(sink)
    stream.on_text(text)
    stream.end()
    return "".join(sink.pieces)
