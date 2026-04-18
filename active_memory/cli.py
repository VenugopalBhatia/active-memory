#!/usr/bin/env python3
"""active-memory CLI — Terminal chat with managed context.

A drop-in replacement for raw API calls that actively manages your
context window using the semantic B-tree. Designed for long coding
and architecture sessions where context rot kills productivity.

Usage:
    python -m active_memory.cli
    python -m active_memory.cli --session myproject
    python -m active_memory.cli --resume

Slash commands inside the chat:
    /stats      Show tree stats (size, depth, tokens)
    /tree       Visualise the B-tree structure
    /hot  [n]   Show the n hottest tuples
    /cold [n]   Show the n coldest tuples
    /prune      Force a prune cycle
    /compress   Force compression of cold subtrees
    /ingest <file>   Ingest a file into the memory tree
    /save       Save session to disk
    /load       Load a previous session
    /export     Export tree contents as JSON
    /budget     Show token budget breakdown
    /help       Show this help
    /quit       Exit
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

# ── ANSI helpers ──────────────────────────────────────────────────────

class C:
    """ANSI colour codes."""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[31m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"
    WHITE   = "\033[37m"
    GREY    = "\033[90m"

    @staticmethod
    def strip(text: str) -> str:
        return re.sub(r'\033\[[0-9;]*m', '', text)


def _bar(value: float, width: int = 20, fill: str = "█", empty: str = "░") -> str:
    filled = int(value * width)
    return fill * filled + empty * (width - filled)


# ── CLI Application ───────────────────────────────────────────────────

SESSION_DIR = Path.home() / ".active-memory" / "sessions"


class ActiveMemoryCLI:
    """Interactive terminal chat with active memory management."""

    def __init__(
        self,
        client: Any,
        embedder: Any,
        session_name: str = "default",
        config: Any = None,
    ) -> None:
        from .middleware import ActiveMemoryMiddleware, MiddlewareConfig

        self.session_name = session_name
        self.session_path = SESSION_DIR / f"{session_name}.pkl"

        self.cfg = config or MiddlewareConfig()
        self.mw = ActiveMemoryMiddleware(client, embedder, self.cfg)
        self._start_time = time.time()

    def _reset_middleware(self) -> None:
        """Recreate middleware so session loads replace, not merge, state."""
        from .middleware import ActiveMemoryMiddleware

        self.mw = ActiveMemoryMiddleware(
            self.mw.client,
            self.mw.embedder,
            self.cfg,
        )

    # ── Main loop ─────────────────────────────────────────────────

    def run(self) -> None:
        """Start the interactive chat loop."""
        self._print_banner()

        while True:
            try:
                user_input = input(f"\n{C.GREEN}{C.BOLD}you ❯ {C.RESET}")
            except (KeyboardInterrupt, EOFError):
                print(f"\n{C.DIM}Goodbye.{C.RESET}")
                break

            user_input = user_input.strip()
            if not user_input:
                continue

            # Handle slash commands
            if user_input.startswith("/"):
                if self._handle_command(user_input):
                    continue
                else:
                    break  # /quit

            # Send through middleware
            try:
                self._print_thinking()
                response = self.mw.send(user_input)
                self._print_response(response.text)
                self._print_status_bar(response.context_stats)
            except Exception as e:
                print(f"\n{C.RED}  Error: {e}{C.RESET}")

    # ── Command handlers ──────────────────────────────────────────

    def _handle_command(self, cmd: str) -> bool:
        """Handle a slash command. Returns True to continue, False to quit."""
        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        handlers = {
            "/help": self._cmd_help,
            "/stats": self._cmd_stats,
            "/tree": self._cmd_tree,
            "/hot": lambda: self._cmd_top(arg, hot=True),
            "/cold": lambda: self._cmd_top(arg, hot=False),
            "/prune": self._cmd_prune,
            "/compress": self._cmd_compress,
            "/ingest": lambda: self._cmd_ingest(arg),
            "/save": self._cmd_save,
            "/load": self._cmd_load,
            "/export": self._cmd_export,
            "/budget": self._cmd_budget,
            "/quit": lambda: None,
            "/q": lambda: None,
        }

        if command in ("/quit", "/q"):
            self._cmd_save()
            print(f"{C.DIM}Session saved. Goodbye.{C.RESET}")
            return False

        handler = handlers.get(command)
        if handler:
            handler()
        else:
            print(f"{C.YELLOW}  Unknown command: {command}. Type /help for options.{C.RESET}")

        return True

    def _cmd_help(self) -> None:
        print(f"""
{C.BOLD}  Commands:{C.RESET}
  {C.CYAN}/stats{C.RESET}          Tree statistics
  {C.CYAN}/tree{C.RESET}           Visualise B-tree structure
  {C.CYAN}/hot [n]{C.RESET}        Show n hottest tuples (default 5)
  {C.CYAN}/cold [n]{C.RESET}       Show n coldest tuples (default 5)
  {C.CYAN}/prune{C.RESET}          Force prune cold tuples
  {C.CYAN}/compress{C.RESET}       Force compress cold subtrees
  {C.CYAN}/ingest <file>{C.RESET}  Ingest a file into memory
  {C.CYAN}/save{C.RESET}           Save session to disk
  {C.CYAN}/load{C.RESET}           Load previous session
  {C.CYAN}/export{C.RESET}         Export tree as JSON
  {C.CYAN}/budget{C.RESET}         Token budget breakdown
  {C.CYAN}/quit{C.RESET}           Save and exit
""")

    def _cmd_stats(self) -> None:
        stats = self.mw.stats
        tree = self.mw.tree
        elapsed = time.time() - self._start_time

        all_tuples = tree.all_tuples()
        total_tokens = sum(t.token_cost for t in all_tuples)
        total_hits = sum(t.hit_count for t in all_tuples)
        nodes = tree.all_nodes()
        compressed = sum(1 for n in nodes if n.summary is not None)

        print(f"""
{C.BOLD}  ┌─ Memory Tree ──────────────────────────────────┐{C.RESET}
  │  Tuples:       {stats['tree_size']:>6}                          │
  │  Nodes:        {stats['total_nodes']:>6}   ({compressed} compressed)       │
  │  Depth:        {stats['tree_depth']:>6}                          │
  │  Total tokens: {total_tokens:>6}                          │
  │  Total hits:   {total_hits:>6}                          │
  │  Turns:        {stats['conversation_turns']:>6}                          │
  │  Session:      {elapsed/60:>5.1f}m                          │
{C.BOLD}  └─────────────────────────────────────────────────┘{C.RESET}
""")

    def _cmd_tree(self) -> None:
        """Print a visual representation of the tree structure."""
        print(f"\n{C.BOLD}  B-Tree Structure:{C.RESET}\n")
        self._print_tree_node(self.mw.tree.root, prefix="  ", is_last=True)
        print()

    def _print_tree_node(
        self, node: Any, prefix: str = "", is_last: bool = True
    ) -> None:
        connector = "└── " if is_last else "├── "

        if node.summary:
            label = f"{C.MAGENTA}◆ compressed{C.RESET} {C.DIM}({node.summary.token_cost} tok){C.RESET}"
        elif node.is_leaf:
            hit_sum = sum(t.hit_count for t in node.tuples)
            label = (
                f"{C.GREEN}● leaf{C.RESET} "
                f"{C.DIM}[{len(node.tuples)} tuples, "
                f"{sum(t.token_cost for t in node.tuples)} tok, "
                f"{hit_sum} hits]{C.RESET}"
            )
        else:
            label = f"{C.BLUE}■ internal{C.RESET} {C.DIM}[{len(node.children)} children]{C.RESET}"

        print(f"{prefix}{connector}{label}")

        child_prefix = prefix + ("    " if is_last else "│   ")

        # Show tuple summaries for leaves
        if node.is_leaf and node.tuples:
            for i, t in enumerate(node.tuples[:6]):
                is_tuple_last = (i == min(len(node.tuples), 6) - 1)
                tc = "└── " if is_tuple_last else "├── "
                age = time.time() - t.last_accessed
                age_str = f"{age:.0f}s" if age < 60 else f"{age/60:.0f}m"

                heat = C.RED if t.hit_count > 5 else C.YELLOW if t.hit_count > 0 else C.GREY
                print(
                    f"{child_prefix}{tc}"
                    f"{heat}⦿{C.RESET} "
                    f"{C.DIM}{t.key_text[:50]}{C.RESET} "
                    f"{heat}[hits={t.hit_count} age={age_str}]{C.RESET}"
                )
            if len(node.tuples) > 6:
                print(f"{child_prefix}    {C.DIM}... +{len(node.tuples) - 6} more{C.RESET}")

        # Recurse into children
        for i, child in enumerate(node.children):
            self._print_tree_node(
                child, child_prefix, is_last=(i == len(node.children) - 1)
            )

    def _cmd_top(self, arg: str, hot: bool) -> None:
        n = int(arg) if arg.strip().isdigit() else 5
        tuples = self.mw.tree.all_tuples()
        if not tuples:
            print(f"  {C.DIM}No tuples in tree.{C.RESET}")
            return

        scored = [
            (self.mw.scorer.score(t), t) for t in tuples
        ]
        scored.sort(key=lambda x: x[0], reverse=hot)
        label = "Hottest" if hot else "Coldest"
        colour = C.RED if hot else C.BLUE

        print(f"\n{C.BOLD}  {label} {n} tuples:{C.RESET}")
        for score, t in scored[:n]:
            age = time.time() - t.last_accessed
            age_str = f"{age:.0f}s" if age < 120 else f"{age/60:.0f}m"
            bar = _bar(score)
            print(
                f"  {colour}{bar}{C.RESET} {score:.3f}  "
                f"{C.DIM}hits={t.hit_count:<3} age={age_str:<6}{C.RESET}  "
                f"{t.key_text[:50]}"
            )
        print()

    def _cmd_prune(self) -> None:
        before = self.mw.tree.size
        evicted = self.mw.tree.prune()
        after = self.mw.tree.size
        print(
            f"  {C.YELLOW}Pruned {len(evicted)} tuples{C.RESET} "
            f"{C.DIM}({before} → {after}){C.RESET}"
        )
        if evicted:
            for t in evicted[:5]:
                print(f"    {C.DIM}✂ {t.key_text[:60]}{C.RESET}")
            if len(evicted) > 5:
                print(f"    {C.DIM}... +{len(evicted) - 5} more{C.RESET}")

    def _cmd_compress(self) -> None:
        before_size = self.mw.tree.size
        count = self.mw.tree.compress_cold_subtrees()
        after_size = self.mw.tree.size
        print(
            f"  {C.MAGENTA}Compressed {count} subtrees{C.RESET} "
            f"{C.DIM}({before_size} → {after_size} tuples){C.RESET}"
        )

    def _cmd_ingest(self, filepath: str) -> None:
        filepath = filepath.strip().strip("'\"")
        if not filepath:
            print(f"  {C.YELLOW}Usage: /ingest <filepath>{C.RESET}")
            return

        path = Path(filepath).expanduser()
        if not path.exists():
            print(f"  {C.RED}File not found: {path}{C.RESET}")
            return

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            tuples = self.mw.ingest_document(text, chunk_size=300)
            total_tokens = sum(t.token_cost for t in tuples)
            print(
                f"  {C.GREEN}Ingested {path.name}{C.RESET}: "
                f"{len(tuples)} chunks, ~{total_tokens} tokens"
            )
        except Exception as e:
            print(f"  {C.RED}Error reading file: {e}{C.RESET}")

    def _cmd_save(self) -> None:
        SESSION_DIR.mkdir(parents=True, exist_ok=True)
        state = {
            "conversation": self.mw._conversation,
            "turn_count": self.mw._turn_count,
            "embedding": {
                "provider": type(self.mw.embedder).__name__,
                "model": getattr(self.mw.embedder, "model", None),
                "dim": int(self.mw.embedder.dim),
            },
            "tree_tuples": [
                {
                    "key_text": t.key_text,
                    "value_text": t.value_text,
                    "key_emb": t.key_emb.tolist() if t.key_emb is not None else None,
                    "id": t.id,
                    "created_at": t.created_at,
                    "last_accessed": t.last_accessed,
                    "hit_count": t.hit_count,
                    "token_cost": t.token_cost,
                    "references": t.references,
                    "referenced_by": t.referenced_by,
                    "tags": t.tags,
                }
                for t in self.mw.tree.all_tuples()
            ],
            "session_name": self.session_name,
            "saved_at": time.time(),
        }
        self.session_path.write_text(json.dumps(state, indent=2))
        print(f"  {C.GREEN}Session saved:{C.RESET} {C.DIM}{self.session_path}{C.RESET}")

    def _cmd_load(self) -> None:
        if not self.session_path.exists():
            # List available sessions
            if SESSION_DIR.exists():
                sessions = list(SESSION_DIR.glob("*.pkl")) + list(SESSION_DIR.glob("*.json"))
                if sessions:
                    print(f"  {C.BOLD}Available sessions:{C.RESET}")
                    for s in sessions:
                        print(f"    {C.CYAN}{s.stem}{C.RESET}")
                else:
                    print(f"  {C.DIM}No saved sessions found.{C.RESET}")
            else:
                print(f"  {C.DIM}No saved sessions found.{C.RESET}")
            return

        try:
            import numpy as np
            from .types import KVTuple

            state = json.loads(self.session_path.read_text())
            self._reset_middleware()
            self.mw._conversation = state["conversation"]
            self.mw._turn_count = state["turn_count"]
            saved_embedding = state.get("embedding", {})
            expected = {
                "provider": type(self.mw.embedder).__name__,
                "model": getattr(self.mw.embedder, "model", None),
                "dim": int(self.mw.embedder.dim),
            }
            tuple_dicts = state["tree_tuples"]
            reembed = bool(tuple_dicts) and (
                not saved_embedding
                or any(
                    saved_embedding.get(key) != expected.get(key)
                    for key in ("provider", "model", "dim")
                )
            )
            reembedded: list[np.ndarray] | None = None
            if reembed:
                keys = [td["key_text"] for td in tuple_dicts]
                reembedded = []
                for start in range(0, len(keys), 256):
                    reembedded.extend(self.mw.embedder.embed(keys[start : start + 256]))

            # Rebuild tree from saved tuples
            for idx, td in enumerate(tuple_dicts):
                t = KVTuple(
                    key_text=td["key_text"],
                    value_text=td["value_text"],
                    key_emb=(
                        reembedded[idx]
                        if reembedded is not None
                        else (
                            np.array(td["key_emb"], dtype=np.float32)
                            if td["key_emb"] else None
                        )
                    ),
                    id=td["id"],
                    created_at=td["created_at"],
                    last_accessed=td["last_accessed"],
                    hit_count=td["hit_count"],
                    token_cost=td["token_cost"],
                    references=td.get("references", []),
                    referenced_by=td.get("referenced_by", []),
                    tags=td.get("tags", []),
                )
                self.mw.tree.insert_tuple(t)

            print(
                f"  {C.GREEN}Session loaded:{C.RESET} "
                f"{len(state['tree_tuples'])} tuples, "
                f"{state['turn_count']} turns"
            )
        except Exception as e:
            print(f"  {C.RED}Error loading session: {e}{C.RESET}")

    def _cmd_export(self) -> None:
        tuples = self.mw.tree.all_tuples()
        export = []
        for t in tuples:
            export.append({
                "key": t.key_text,
                "value": t.value_text[:200],
                "hits": t.hit_count,
                "age_seconds": round(time.time() - t.last_accessed, 1),
                "tokens": t.token_cost,
                "score": round(self.mw.scorer.score(t), 4),
            })
        export.sort(key=lambda x: x["score"], reverse=True)

        path = Path(f"active_memory_export_{self.session_name}.json")
        path.write_text(json.dumps(export, indent=2))
        print(f"  {C.GREEN}Exported {len(export)} tuples to {path}{C.RESET}")

    def _cmd_budget(self) -> None:
        cfg = self.mw.assembler.cfg
        tree = self.mw.tree
        all_tuples = tree.all_tuples()
        total_stored = sum(t.token_cost for t in all_tuples)
        conv_tokens = sum(
            len(m.get("content", "")) // 4 for m in self.mw._conversation
        )

        budget = cfg.total_budget
        used_ratio = min(1.0, total_stored / max(1, budget))
        conv_ratio = min(1.0, conv_tokens / max(1, budget))

        print(f"""
{C.BOLD}  Token Budget Breakdown{C.RESET}
  ─────────────────────────────────────────
  Total budget:       {budget:>8,} tokens
  Pinned reserve:     {cfg.pinned_reserve:>8,} tokens
  Recency window:     {cfg.recency_window:>8} turns

  Raw conversation:   {conv_tokens:>8,} tokens
  Stored in tree:     {total_stored:>8,} tokens
  Tree compression:   {C.GREEN}{max(0, conv_tokens - total_stored):>8,}{C.RESET} tokens saved

  {C.DIM}Stored  {C.RESET} {_bar(used_ratio)} {used_ratio:.0%} of budget
  {C.DIM}Raw conv{C.RESET} {_bar(conv_ratio)} {conv_ratio:.0%} of budget
""")

    # ── Display helpers ───────────────────────────────────────────

    def _print_banner(self) -> None:
        print(f"""
{C.BOLD}{C.CYAN}  ╔═══════════════════════════════════════════╗
  ║         active-memory v0.1.0               ║
  ║    Context-managed LLM conversations       ║
  ╚═══════════════════════════════════════════╝{C.RESET}
  {C.DIM}Session: {self.session_name}
  Model:   {self.cfg.model}
  Budget:  {self.cfg.assembler.total_budget:,} tokens
  Type /help for commands{C.RESET}
""")

    def _print_thinking(self) -> None:
        print(f"  {C.DIM}thinking...{C.RESET}", end="", flush=True)

    def _print_response(self, text: str) -> None:
        # Clear the "thinking..." line
        print(f"\r{' ' * 40}\r", end="")
        print(f"\n{C.CYAN}{C.BOLD}claude ❯{C.RESET} {text}\n")

    def _print_status_bar(self, stats: Any) -> None:
        tree_size = stats.tree_size
        depth = stats.tree_depth
        tokens = stats.tokens_assembled
        budget = self.mw.assembler.cfg.total_budget
        usage = min(1.0, tokens / max(1, budget))

        # Colour the usage bar based on pressure
        if usage > 0.8:
            bar_colour = C.RED
        elif usage > 0.5:
            bar_colour = C.YELLOW
        else:
            bar_colour = C.GREEN

        bar = _bar(usage, width=15)

        print(
            f"  {C.DIM}─── "
            f"tree: {tree_size} tuples │ "
            f"depth: {depth} │ "
            f"turn: {stats.turn} │ "
            f"context: {bar_colour}{bar}{C.RESET}{C.DIM} "
            f"{tokens:,}/{budget:,} tok "
            f"({stats.tuples_included}/{stats.tuples_considered} tuples)"
            f" ───{C.RESET}"
        )


# ── Entry point ───────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="active-memory: Terminal chat with managed context",
    )
    parser.add_argument(
        "--session", "-s",
        default="default",
        help="Session name for save/load (default: 'default')",
    )
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        help="Resume the last saved session",
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "ollama", "gemini", "codex"],
        default="anthropic",
        help="Model provider (default: anthropic). 'ollama' uses local Ollama server",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Model name (default: provider-dependent)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Custom API base URL for OpenAI-compatible servers (Ollama, LM Studio, vLLM)",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=100_000,
        help="Total token budget (default: 100000)",
    )
    parser.add_argument(
        "--embedder",
        choices=["auto", "hash", "openai", "local", "gemini"],
        default="auto",
        help="Embedding provider (default: auto). Prefers OpenAI > Gemini > local > hash",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=64,
        help="Embedding dimension for hash embedder (default: 64)",
    )
    parser.add_argument(
        "--embed-model",
        default="text-embedding-3-small",
        help="OpenAI embedding model when using auto/openai (default: text-embedding-3-small)",
    )
    parser.add_argument(
        "--ingest",
        nargs="*",
        help="Files to ingest at startup",
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Path to JSON config file (default: ~/.active-memory/config.json if it exists)",
    )

    args = parser.parse_args()

    # -- Resolve default model per provider --
    from .model_clients import DEFAULT_PROVIDER_MODELS, PROVIDER_AUTH_ENV_VARS, PROVIDER_PACKAGE_EXTRAS

    if args.model is None:
        args.model = DEFAULT_PROVIDER_MODELS.get(args.provider, "gpt-4o-mini")

    # -- Set up model client --
    from .model_clients import create_model_client

    try:
        client = create_model_client(args.provider, base_url=args.base_url)
    except ImportError:
        pkg = PROVIDER_PACKAGE_EXTRAS.get(args.provider, args.provider)
        print(f"{C.RED}Error: '{pkg}' package not installed.{C.RESET}")
        print(f"{C.DIM}  pip install active-memory[{pkg}]{C.RESET}")
        sys.exit(1)
    except RuntimeError as exc:
        print(f"{C.RED}Error: {exc}.{C.RESET}")
        if args.provider == "anthropic":
            print(f"{C.DIM}  Option 1: export ANTHROPIC_API_KEY=sk-ant-...{C.RESET}")
            print(f"{C.DIM}  Option 2: Log into Claude Code (claude) — the chat CLI will reuse your session{C.RESET}")
        elif args.provider == "ollama":
            print(f"{C.DIM}  Make sure Ollama is running: ollama serve{C.RESET}")
        else:
            env_var = PROVIDER_AUTH_ENV_VARS.get(args.provider, "OPENAI_API_KEY")
            print(f"{C.DIM}  export {env_var}=...{C.RESET}")
        sys.exit(1)

    # -- Set up embedder --
    from .embeddings import create_embedder

    embedder_spec = create_embedder(
        args.embedder,
        dim=args.embed_dim,
        openai_model=args.embed_model,
        verbose=True,
        stream=sys.stderr,
    )
    embedder = embedder_spec.embedder
    colour = C.GREEN if embedder_spec.semantic else C.YELLOW
    print(f"{colour}  Using {embedder_spec.description}{C.RESET}")
    provider_label = args.provider
    if args.base_url:
        provider_label += f" @ {args.base_url}"
    # Show auth mode for Anthropic
    if args.provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        provider_label += " (Claude Code session)"
    print(f"{C.CYAN}  Model provider: {provider_label}{C.RESET}")

    # -- Build config (defaults < config file < CLI flags) --
    from .config import load_config, build_middleware_config

    file_overrides = load_config(args.config)
    config = build_middleware_config(file_overrides)

    # CLI flags override config file when explicitly set
    if _user_passed_model:
        config.model = args.model
    elif "model" not in file_overrides:
        config.model = args.model  # provider default
    if args.budget != 100_000:
        config.assembler.total_budget = args.budget

    # -- Create CLI --
    cli = ActiveMemoryCLI(
        client=client,
        embedder=embedder,
        session_name=args.session,
        config=config,
    )

    # -- Resume if requested --
    if args.resume:
        cli._cmd_load()

    # -- Ingest files if provided --
    if args.ingest:
        for filepath in args.ingest:
            cli._cmd_ingest(filepath)

    # -- Run --
    cli.run()


if __name__ == "__main__":
    main()
