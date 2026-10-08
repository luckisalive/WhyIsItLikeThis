"""
Why The Code Is Like This — CLI Entry Point.

Usage:
    python main.py setup              # Initialize Neo4j schema
    python main.py ingest             # Run full ingestion pipeline
    python main.py ingest --quick     # Run quick demo ingestion
    python main.py query "question"   # Ask a question
    python main.py stats              # Show graph statistics
    python main.py check-rate-limit   # Check GitHub API rate limit
"""

import argparse
import asyncio
import logging
import sys

# Ensure UTF-8 output on Windows consoles to prevent UnicodeEncodeError with emojis
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from dotenv import load_dotenv

# Load .env before importing settings
load_dotenv()

from settings import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def cmd_setup(args):
    """Initialize the Neo4j graph schema (constraints + indexes)."""
    settings = get_settings()
    from src.graph.neo4j_manager import Neo4jManager

    logger.info("Connecting to Neo4j AuraDB...")
    mgr = Neo4jManager(settings.neo4j_uri, settings.neo4j_username, settings.neo4j_password)
    try:
        mgr.setup_schema()
        logger.info("✅ Schema setup complete. Constraints and indexes created.")
    finally:
        mgr.close()


def cmd_ingest(args):
    """Run the ingestion pipeline."""
    settings = get_settings()

    if args.quick:
        settings.max_commits = args.max_commits or 50
        settings.max_prs = args.max_prs or 30
        settings.max_issues = args.max_issues or 20
        logger.info(
            "🚀 Quick mode: max_commits=%d, max_prs=%d, max_issues=%d",
            settings.max_commits,
            settings.max_prs,
            settings.max_issues,
        )
    else:
        if args.max_commits:
            settings.max_commits = args.max_commits
        if args.max_prs:
            settings.max_prs = args.max_prs
        if args.max_issues:
            settings.max_issues = args.max_issues

    from src.pipeline import IngestionPipeline

    pipeline = IngestionPipeline(settings)

    def progress(msg, pct):
        bar_len = 30
        filled = int(bar_len * pct)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\r  [{bar}] {pct*100:.0f}% — {msg}", end="", flush=True)
        if pct >= 1.0:
            print()

    logger.info("Starting ingestion for repository: %s", settings.target_repo)
    asyncio.run(pipeline.run(progress_callback=progress))
    logger.info("✅ Ingestion complete.")


def cmd_query(args):
    """Ask a question and get an evidence-backed answer."""
    question = " ".join(args.question)
    if not question.strip():
        logger.error("Please provide a question.")
        sys.exit(1)

    settings = get_settings()
    from src.graph.neo4j_manager import Neo4jManager
    from src.graph.embedding_generator import EmbeddingGenerator
    from src.query.retriever import WhyRetriever
    from src.query.answer_engine import AnswerEngine

    logger.info("Connecting to Neo4j and initializing query engine...")
    neo4j_mgr = Neo4jManager(settings.neo4j_uri, settings.neo4j_username, settings.neo4j_password)
    embedding_gen = EmbeddingGenerator(
        settings.google_api_key, settings.embedding_model, settings.embedding_dimensions
    )
    retriever = WhyRetriever(neo4j_mgr, embedding_gen)
    groq_fallbacks = [m.strip() for m in settings.groq_fallback_models.split(",") if m.strip()]
    gemini_reasoning = [m.strip() for m in settings.gemini_fallback_models.split(",") if m.strip()]
    engine = AnswerEngine(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        retriever=retriever,
        fallback_models=groq_fallbacks,
        google_api_key=settings.google_api_key,
        gemini_reasoning_models=gemini_reasoning,
        enable_cross_provider_fallback=settings.enable_cross_provider_fallback,
        cooldown_seconds=settings.rate_limit_cooldown_seconds,
    )



    try:
        logger.info("🔍 Searching for: %s", question)
        answer = engine.answer(question)

        # Display answer
        print("\n" + "=" * 70)
        print(f"❓ Question: {answer.question}")
        print("=" * 70)

        # Confidence badge
        badge = {"HIGH": "🟢 HIGH", "MEDIUM": "🟡 MEDIUM", "LOW": "🔴 LOW"}
        print(f"\n📊 Confidence: {badge.get(answer.confidence.value, answer.confidence.value)}")
        print(f"   {answer.confidence_explanation}")

        # Answer
        print(f"\n💡 Answer:\n{answer.answer_text}")

        # Evidence
        if answer.evidence:
            print(f"\n📎 Evidence ({len(answer.evidence)} items):")
            for i, ev in enumerate(answer.evidence, 1):
                date_str = ev.date.strftime("%Y-%m-%d") if ev.date else "unknown"
                print(f"  {i}. [{ev.source_type}] {ev.title}")
                print(f"     Date: {date_str} | By: {ev.person or 'unknown'}")
                if ev.url:
                    print(f"     URL: {ev.url}")
                print(f"     {ev.summary}")

        # Supersession
        if answer.supersession:
            print(f"\n⚠️  Supersession:")
            print(f"   '{answer.supersession.superseded_decision}' was later overridden by")
            print(f"   '{answer.supersession.superseded_by}'")
            print(f"   Reason: {answer.supersession.reason}")

        # Ask person
        if answer.ask_person:
            print(f"\n👤 Talk to: @{answer.ask_person}")

        print("\n" + "=" * 70)

    finally:
        neo4j_mgr.close()


def cmd_stats(args):
    """Show graph statistics."""
    settings = get_settings()
    from src.graph.neo4j_manager import Neo4jManager

    neo4j_mgr = Neo4jManager(settings.neo4j_uri, settings.neo4j_username, settings.neo4j_password)
    try:
        stats = neo4j_mgr.get_stats()
        print("\n📊 Knowledge Graph Statistics")
        print("=" * 40)
        print(f"  Decisions (total):     {stats.total_decisions}")
        print(f"    ├─ Active:           {stats.active_decisions}")
        print(f"    └─ Superseded:       {stats.superseded_decisions}")
        print(f"  Commits:               {stats.total_commits}")
        print(f"  Pull Requests:         {stats.total_prs}")
        print(f"  Issues:                {stats.total_issues}")
        print(f"  People:                {stats.total_people}")
        print(f"  Code Entities:         {stats.total_code_entities}")
        print(f"  Relationships:         {stats.total_relationships}")
        print("=" * 40)
    finally:
        neo4j_mgr.close()


def cmd_check_rate_limit(args):
    """Check GitHub API rate limit status."""
    settings = get_settings()
    from github import Github, Auth

    g = Github(auth=Auth.Token(settings.github_token))
    rl = g.get_rate_limit()
    rate = getattr(rl, "rate", getattr(rl, "core", None))
    search_rate = getattr(rl, "search", None)
    print("\n🔑 GitHub API Rate Limit Status")
    print("=" * 40)
    if rate:
        print(f"  Core:   {rate.remaining} / {rate.limit}")
        print(f"  Resets: {rate.reset.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    if search_rate:
        print(f"  Search: {search_rate.remaining} / {search_rate.limit}")
    print("=" * 40)
    g.close()


def main():
    parser = argparse.ArgumentParser(
        prog="why-the-code",
        description="Why The Code Is Like This — Answer why questions about any codebase.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # setup
    subparsers.add_parser("setup", help="Initialize Neo4j graph schema")

    # ingest
    ingest_parser = subparsers.add_parser("ingest", help="Run ingestion pipeline")
    ingest_parser.add_argument("--quick", action="store_true", help="Quick demo mode (fewer items)")
    ingest_parser.add_argument("--max-commits", type=int, help="Max commits to ingest")
    ingest_parser.add_argument("--max-prs", type=int, help="Max pull requests to ingest")
    ingest_parser.add_argument("--max-issues", type=int, help="Max issues to ingest")

    # query
    query_parser = subparsers.add_parser("query", help="Ask a question")
    query_parser.add_argument("question", nargs="+", help="The question to ask")

    # stats
    subparsers.add_parser("stats", help="Show graph statistics")

    # check-rate-limit
    subparsers.add_parser("check-rate-limit", help="Check GitHub API rate limit")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "setup": cmd_setup,
        "ingest": cmd_ingest,
        "query": cmd_query,
        "stats": cmd_stats,
        "check-rate-limit": cmd_check_rate_limit,
    }

    try:
        commands[args.command](args)
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
        sys.exit(130)
    except Exception as e:
        logger.error("❌ Error: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
