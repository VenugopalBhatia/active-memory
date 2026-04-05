"""AST-aware code ingestion and call graph analysis.

Inspired by codingAgent's approach: instead of splitting code on sentence
boundaries (which breaks at every `self.`, `np.`, etc.), we parse the
AST and extract functions, classes, and methods as discrete chunks.

We also extract the call graph so the scoring engine can boost
structurally related tuples — if function A calls function B, querying
for A should also surface B even if their names are semantically
unrelated.
"""

from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CodeChunk:
    """A discrete unit of code extracted from an AST node."""
    name: str                    # e.g. "MyClass.my_method" or "helper_func"
    kind: str                    # "function", "method", "class", "module_code"
    source: str                  # the raw source code
    filepath: str                # originating file
    line_start: int
    line_end: int
    calls: list[str] = field(default_factory=list)       # names this chunk calls
    called_by: list[str] = field(default_factory=list)    # names that call this chunk
    decorators: list[str] = field(default_factory=list)
    docstring: str | None = None


class CodeParser:
    """Parses Python source into AST-based chunks with call graph edges."""

    def parse_file(self, filepath: str | Path) -> list[CodeChunk]:
        """Parse a Python file into code chunks."""
        filepath = Path(filepath)
        try:
            source = filepath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return []
        return self.parse_source(source, str(filepath))

    def parse_source(self, source: str, filepath: str = "<string>") -> list[CodeChunk]:
        """Parse Python source code into code chunks with call graph."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            # Fall back to line-based chunking for unparseable files
            return self._fallback_chunk(source, filepath)

        lines = source.splitlines()
        chunks: list[CodeChunk] = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                chunks.extend(self._extract_class(node, lines, filepath))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunks.append(self._extract_function(node, lines, filepath))

        # Extract module-level code (imports, constants, assignments)
        module_lines = self._extract_module_level(tree, lines)
        if module_lines:
            chunks.insert(0, CodeChunk(
                name="__module__",
                kind="module_code",
                source=module_lines,
                filepath=filepath,
                line_start=1,
                line_end=len(lines),
            ))

        # Build call graph edges
        self._resolve_call_graph(chunks)

        return chunks

    def _extract_class(
        self, node: ast.ClassDef, lines: list[str], filepath: str
    ) -> list[CodeChunk]:
        """Extract a class and its methods as separate chunks."""
        chunks: list[CodeChunk] = []

        # Class-level chunk (docstring + class vars, no method bodies)
        class_header = self._get_source_segment(node, lines)
        decorators = [self._decorator_name(d) for d in node.decorator_list]
        docstring = ast.get_docstring(node)

        # Extract each method
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_chunk = self._extract_function(
                    child, lines, filepath,
                    class_name=node.name,
                )
                chunks.append(method_chunk)

        # Also create a chunk for the class signature + docstring
        class_sig_lines = []
        for deco in node.decorator_list:
            class_sig_lines.append(self._get_source_segment(deco, lines))
        class_sig_lines.append(f"class {node.name}:")
        if docstring:
            class_sig_lines.append(f'    """{docstring}"""')

        # Add class-level assignments
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.Assign, ast.AnnAssign)):
                class_sig_lines.append("    " + self._get_source_segment(child, lines))

        if class_sig_lines:
            chunks.insert(0, CodeChunk(
                name=node.name,
                kind="class",
                source="\n".join(class_sig_lines),
                filepath=filepath,
                line_start=node.lineno,
                line_end=node.end_lineno or node.lineno,
                decorators=decorators,
                docstring=docstring,
            ))

        return chunks

    def _extract_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        lines: list[str],
        filepath: str,
        class_name: str | None = None,
    ) -> CodeChunk:
        """Extract a function or method as a chunk."""
        name = f"{class_name}.{node.name}" if class_name else node.name
        kind = "method" if class_name else "function"
        source = self._get_source_segment(node, lines)
        calls = self._extract_calls(node)
        decorators = [self._decorator_name(d) for d in node.decorator_list]
        docstring = ast.get_docstring(node)

        return CodeChunk(
            name=name,
            kind=kind,
            source=source,
            filepath=filepath,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            calls=calls,
            decorators=decorators,
            docstring=docstring,
        )

    def _extract_calls(self, node: ast.AST) -> list[str]:
        """Extract all function/method names called within an AST node."""
        calls: list[str] = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    # self.method() → "method"
                    # obj.method() → "method"
                    calls.append(child.func.attr)
                    # Also record "obj.method" for better matching
                    if isinstance(child.func.value, ast.Name):
                        calls.append(f"{child.func.value.id}.{child.func.attr}")
        return list(set(calls))

    def _extract_module_level(
        self, tree: ast.Module, lines: list[str]
    ) -> str:
        """Extract module-level statements (imports, constants)."""
        parts: list[str] = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                parts.append(self._get_source_segment(node, lines))
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                parts.append(self._get_source_segment(node, lines))
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                # Module docstring
                if isinstance(node.value.value, str):
                    parts.append(self._get_source_segment(node, lines))
        return "\n".join(parts) if parts else ""

    def _resolve_call_graph(self, chunks: list[CodeChunk]) -> None:
        """Build bidirectional call graph edges between chunks."""
        name_to_chunk: dict[str, CodeChunk] = {}
        for chunk in chunks:
            name_to_chunk[chunk.name] = chunk
            # Also register by short name for method matching
            if "." in chunk.name:
                short = chunk.name.split(".")[-1]
                # Only register if not ambiguous
                if short not in name_to_chunk:
                    name_to_chunk[short] = chunk

        for chunk in chunks:
            for call_name in chunk.calls:
                target = name_to_chunk.get(call_name)
                if target and target.name != chunk.name:
                    if chunk.name not in target.called_by:
                        target.called_by.append(chunk.name)

    @staticmethod
    def _get_source_segment(node: ast.AST, lines: list[str]) -> str:
        """Extract source lines for an AST node."""
        start = getattr(node, "lineno", 1) - 1
        end = getattr(node, "end_lineno", start + 1)
        segment = lines[start:end]
        if not segment:
            return ""
        # Dedent to remove common indentation
        return textwrap.dedent("\n".join(segment))

    @staticmethod
    def _decorator_name(node: ast.expr) -> str:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                return node.func.id
            elif isinstance(node.func, ast.Attribute):
                return node.func.attr
        return "unknown"

    @staticmethod
    def _fallback_chunk(source: str, filepath: str) -> list[CodeChunk]:
        """Line-based fallback for files that don't parse."""
        lines = source.splitlines()
        chunk_size = 50
        chunks = []
        for i in range(0, len(lines), chunk_size):
            block = "\n".join(lines[i:i + chunk_size])
            chunks.append(CodeChunk(
                name=f"block_{i // chunk_size}",
                kind="raw",
                source=block,
                filepath=filepath,
                line_start=i + 1,
                line_end=min(i + chunk_size, len(lines)),
            ))
        return chunks


# ── Multi-language fallback ───────────────────────────────────────────

# File extensions that get AST parsing
AST_PARSEABLE = {".py", ".pyw"}

# Extensions that get structure-aware regex parsing
BRACE_LANGUAGES = {".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs", ".c", ".cpp", ".h"}


def parse_code_file(filepath: str | Path) -> list[CodeChunk]:
    """Parse a code file into chunks. Uses AST for Python, falls
    back to brace-matching for other languages, and line-based
    chunking as a last resort."""
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    if suffix in AST_PARSEABLE:
        return CodeParser().parse_file(filepath)
    elif suffix in BRACE_LANGUAGES:
        return _brace_language_chunks(filepath)
    else:
        # Generic line-based chunking
        try:
            source = filepath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return []
        return CodeParser._fallback_chunk(source, str(filepath))


def _brace_language_chunks(filepath: Path) -> list[CodeChunk]:
    """Regex-based chunking for brace languages.
    Splits on function/class signatures."""
    import re

    source = filepath.read_text(encoding="utf-8", errors="replace")
    lines = source.splitlines()

    # Pattern matches common function/method/class declarations
    pattern = re.compile(
        r'^(?:(?:export\s+)?(?:async\s+)?(?:function|class|def|fn|func|pub\s+fn|pub\s+func)\s+\w+|'
        r'(?:public|private|protected|static)\s+.*?\w+\s*\()',
        re.MULTILINE
    )

    # Find all declaration starts
    starts: list[int] = []
    for match in pattern.finditer(source):
        lineno = source[:match.start()].count("\n")
        starts.append(lineno)

    if not starts:
        return CodeParser._fallback_chunk(source, str(filepath))

    chunks: list[CodeChunk] = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(lines)
        block = "\n".join(lines[start:end]).rstrip()
        # Extract name from first line
        first_line = lines[start].strip()
        name_match = re.search(r'\b(\w+)\s*[({]', first_line)
        name = name_match.group(1) if name_match else f"block_{i}"

        chunks.append(CodeChunk(
            name=name,
            kind="function",
            source=block,
            filepath=str(filepath),
            line_start=start + 1,
            line_end=end,
        ))

    return chunks
