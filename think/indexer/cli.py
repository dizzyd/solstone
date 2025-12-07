"""CLI functionality for the indexer."""

import argparse
import os
from typing import Any

from think.utils import journal_log, setup_cli

from .core import reset_index
from .events import scan_events, search_events
from .insights import scan_insights, search_insights
from .journal import (
    reset_journal_index,
    scan_journal,
    search_journal,
)
from .transcripts import scan_transcripts, search_transcripts


def _display_search_results(results: list[dict[str, Any]]) -> None:
    """Display search results in a consistent format."""
    for idx, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        snippet = (
            r["text"][:100] + "..."
            if len(r.get("text", "")) > 100
            else r.get("text", "")
        )
        label = meta.get("topic") or meta.get("time") or ""
        facet = meta.get("facet")
        facet_str = f" ({facet})" if facet else ""
        print(f"{idx}. {meta.get('day')} {label}{facet_str}: {snippet}")


def main() -> None:
    """Main CLI entry point for the indexer."""
    parser = argparse.ArgumentParser(
        description="Index journal content (insights, transcripts, events, entities, todos)"
    )
    parser.add_argument(
        "--rescan",
        action="store_true",
        help="Scan journal and update the index",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Alias for --rescan (backward compatibility)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Remove the index before rescan",
    )
    parser.add_argument(
        "--day",
        help="Filter search results by YYYYMMDD day",
    )
    parser.add_argument(
        "--facet",
        help="Filter search results by facet name",
    )
    parser.add_argument(
        "--topic",
        help="Filter search results by topic (e.g., 'flow', 'audio', 'event')",
    )
    parser.add_argument(
        "-q",
        "--query",
        nargs="?",
        const="",
        help="Run query (interactive mode if no query provided)",
    )
    # Legacy arguments for backward compatibility
    parser.add_argument(
        "--index",
        choices=["insights", "events", "transcripts"],
        help="(Legacy) Operate on old separate indexes",
    )
    parser.add_argument(
        "--rescan-all",
        action="store_true",
        help="(Legacy) Rescan all old indexes",
    )
    parser.add_argument(
        "--segment",
        help="(Legacy) Limit transcript rescan to segment",
    )
    parser.add_argument(
        "--source",
        help="(Legacy) Filter transcript results by source",
    )

    args = setup_cli(parser)

    # Handle --full as alias for --rescan
    if args.full:
        args.rescan = True

    journal = os.getenv("JOURNAL_PATH")

    # Legacy mode: if --index or --rescan-all specified, use old indexes
    if args.index or args.rescan_all:
        _run_legacy_mode(args, journal)
        return

    # New unified journal index mode
    if not args.rescan and not args.reset and args.query is None:
        parser.print_help()
        return

    if args.reset:
        reset_journal_index(journal)

    if args.rescan:
        changed = scan_journal(journal, verbose=args.verbose)
        if changed:
            journal_log("indexer journal rescan ok")

    if args.query is not None:
        query_kwargs: dict[str, Any] = {}
        if args.day:
            query_kwargs["day"] = args.day
        if args.facet:
            query_kwargs["facet"] = args.facet
        if args.topic:
            query_kwargs["topic"] = args.topic

        if args.query:
            # Single query mode
            _total, results = search_journal(args.query, 10, **query_kwargs)
            _display_search_results(results)
        else:
            # Interactive mode
            while True:
                try:
                    query = input("search> ").strip()
                except EOFError:
                    break
                if not query:
                    break
                _total, results = search_journal(query, 10, **query_kwargs)
                _display_search_results(results)


def _run_legacy_mode(args: argparse.Namespace, journal: str) -> None:
    """Run legacy index operations for backward compatibility."""
    # Validate legacy args
    if args.segment and not args.day:
        raise SystemExit("--segment requires --day to be specified")

    if args.reset and args.index:
        reset_index(
            journal, args.index, day=args.day if args.index == "transcripts" else None
        )

    if args.rescan_all:
        for index_name in ["insights", "events", "transcripts"]:
            if index_name == "transcripts":
                changed = scan_transcripts(journal, verbose=args.verbose)
            elif index_name == "events":
                changed = scan_events(journal, verbose=args.verbose)
            else:
                changed = scan_insights(journal, verbose=args.verbose)
            if changed:
                journal_log(f"indexer {index_name} rescan ok")

    if args.rescan and args.index:
        if args.index == "transcripts":
            changed = scan_transcripts(
                journal, verbose=args.verbose, day=args.day, segment=args.segment
            )
        elif args.index == "events":
            changed = scan_events(journal, verbose=args.verbose)
        else:
            changed = scan_insights(journal, verbose=args.verbose)
        if changed:
            journal_log(f"indexer {args.index} rescan ok")

    if args.query is not None and args.index:
        if args.index == "transcripts":
            search_func = search_transcripts
            query_kwargs: dict[str, Any] = {"day": args.day, "source": args.source}
        elif args.index == "events":
            search_func = search_events
            query_kwargs = {}
        else:
            search_func = search_insights
            query_kwargs = {}

        if args.query:
            _total, results = search_func(args.query, 5, **query_kwargs)
            _display_search_results(results)
        else:
            while True:
                try:
                    query = input("search> ").strip()
                except EOFError:
                    break
                if not query:
                    break
                _total, results = search_func(query, 5, **query_kwargs)
                _display_search_results(results)
