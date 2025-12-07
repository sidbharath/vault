#!/usr/bin/env python3
"""
Vault - Personal Knowledge Base powered by Parallax

A fully local, privacy-first knowledge base that uses Parallax
for distributed AI inference. Index your documents and chat with
them using local LLMs.

Usage:
    python main.py ingest <path>     # Ingest documents
    python main.py chat              # Start interactive chat
    python main.py web               # Start web interface
    python main.py search <query>    # Search documents
    python main.py stats             # Show statistics
    python main.py briefing          # Get your daily briefing
    python main.py sync gmail        # Sync emails from Gmail
    python main.py sync drive        # Sync documents from Drive
"""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Vault - Personal Knowledge Base powered by Parallax",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py ingest ~/Documents/notes     # Index a directory
  python main.py ingest paper.pdf             # Index a single file
  python main.py chat                          # Start chatting
  python main.py web                           # Start web UI at localhost:8080
  python main.py search "machine learning"     # Search your documents
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents into the knowledge base")
    ingest_parser.add_argument("path", help="Path to file or directory to ingest")

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start interactive chat session")
    chat_parser.add_argument(
        "--url", default="http://localhost:3001",
        help="Parallax server URL (default: http://localhost:3001)"
    )

    # Web command
    web_parser = subparsers.add_parser("web", help="Start web interface")
    web_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    web_parser.add_argument("--port", type=int, default=8080, help="Port to listen on")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search indexed documents")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("-n", "--num-results", type=int, default=5, help="Number of results")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show knowledge base statistics")

    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear all indexed documents")

    # Briefing command
    briefing_parser = subparsers.add_parser("briefing", help="Get your daily Chief of Staff briefing")

    # Sync command with subcommands
    sync_parser = subparsers.add_parser("sync", help="Sync external data sources")
    sync_subparsers = sync_parser.add_subparsers(dest="sync_source", help="Data source to sync")

    gmail_sync = sync_subparsers.add_parser("gmail", help="Sync emails from Gmail")
    gmail_sync.add_argument("--days", type=int, default=30, help="Days of email history to sync")
    gmail_sync.add_argument("--max", type=int, default=500, help="Maximum messages to sync")
    gmail_sync.add_argument("--query", default="", help="Gmail search query filter")

    drive_sync = sync_subparsers.add_parser("drive", help="Sync documents from Google Drive")
    drive_sync.add_argument("--days", type=int, default=30, help="Sync files modified in last N days")
    drive_sync.add_argument("--max", type=int, default=200, help="Maximum files to sync")
    drive_sync.add_argument("--folder", default=None, help="Specific folder ID to sync")

    args = parser.parse_args()

    if args.command == "web":
        from src.web import run_server
        print(f"\nStarting Vault web interface at http://localhost:{args.port}")
        print("Press Ctrl+C to stop\n")
        run_server(host=args.host, port=args.port)

    elif args.command == "ingest":
        from src.cli import ingest_documents, show_stats, print_banner
        from src.vectorstore import VectorStore
        from rich.console import Console

        console = Console()
        print_banner()
        vectorstore = VectorStore()
        console.print(f"\n[bold]Ingesting documents from:[/bold] {args.path}\n")
        count = ingest_documents(args.path, vectorstore)
        console.print(f"\n[green]Successfully indexed {count} chunks[/green]")
        show_stats(vectorstore)

    elif args.command == "chat":
        from src.cli import interactive_chat, print_banner
        from src.vectorstore import VectorStore
        from src.rag import RAGEngine

        print_banner()
        vectorstore = VectorStore()
        rag = RAGEngine(vectorstore, parallax_url=args.url)
        interactive_chat(rag)

    elif args.command == "search":
        from src.vectorstore import VectorStore
        from rich.console import Console
        from rich.panel import Panel
        from pathlib import Path

        console = Console()
        vectorstore = VectorStore()
        results = vectorstore.search(args.query, n_results=args.num_results)

        if results:
            console.print(f"\n[bold]Top {len(results)} results for:[/bold] {args.query}\n")
            for i, result in enumerate(results, 1):
                source = Path(result['source']).name
                distance = result.get('distance', 'N/A')
                title = f"[{i}] {source}"
                if isinstance(distance, float):
                    title += f" (similarity: {1-distance:.2%})"
                console.print(Panel(
                    result['content'][:500] + "..." if len(result['content']) > 500 else result['content'],
                    title=title
                ))
        else:
            console.print("[yellow]No results found.[/yellow]")

    elif args.command == "stats":
        from src.cli import show_stats, print_banner
        from src.vectorstore import VectorStore

        print_banner()
        vectorstore = VectorStore()
        show_stats(vectorstore)

    elif args.command == "clear":
        from src.vectorstore import VectorStore
        from rich.console import Console

        console = Console()
        if console.input("[red]Clear all indexed documents? (yes/no): [/red]").lower() == "yes":
            vectorstore = VectorStore()
            vectorstore.clear()
            console.print("[green]Knowledge base cleared.[/green]")
        else:
            console.print("[dim]Cancelled.[/dim]")

    elif args.command == "briefing":
        from src.vectorstore import VectorStore
        from src.rag import RAGEngine
        from src.cli import print_banner
        from rich.console import Console
        from rich.panel import Panel
        from datetime import datetime

        console = Console()
        print_banner()

        vectorstore = VectorStore()
        rag = RAGEngine(vectorstore)

        if not rag.check_parallax():
            console.print("[red]Parallax not running. Start it with: parallax run[/red]")
            sys.exit(1)

        today = datetime.now().strftime("%A, %B %d, %Y")

        console.print(f"\n[bold cyan]🎯 Generating your briefing for {today}...[/bold cyan]\n")

        briefing_prompt = f"""Today is {today}. Based on all my documents, emails, meeting notes, and project information, give me a daily briefing that includes:

1. What are my top priorities for today?
2. What deadlines are coming up this week?
3. Are there any emails I need to respond to?
4. What's the status of my active projects?

Be specific with names, dates, and action items. Format it clearly."""

        # Stream the response
        console.print(Panel.fit("[bold]Daily Briefing[/bold]", border_style="cyan"))
        console.print()

        for chunk in rag.query(briefing_prompt, stream=True, max_tokens=1500):
            console.print(chunk, end="")

        console.print("\n")

    elif args.command == "sync":
        from src.vectorstore import VectorStore
        from src.ingestion import DocumentLoader
        from rich.console import Console

        console = Console()

        if args.sync_source == "gmail":
            from src.integrations.gmail import get_gmail_client

            console.print("\n[bold cyan]📧 Gmail Sync[/bold cyan]\n")

            client = get_gmail_client()

            if not client.is_available():
                console.print("[red]Google API libraries not installed.[/red]")
                console.print("Run: [cyan]pip install google-auth-oauthlib google-api-python-client[/cyan]")
                sys.exit(1)

            if not client.is_configured():
                console.print("[red]Gmail not configured.[/red]")
                console.print(f"Please save your OAuth credentials to: [cyan]~/.vault/credentials.json[/cyan]")
                sys.exit(1)

            console.print("Authenticating with Gmail...")
            client.authenticate()
            profile = client.get_profile()
            console.print(f"[green]Connected as:[/green] {profile['email']}")

            vectorstore = VectorStore()
            loader = DocumentLoader()

            count = client.sync_to_vault(
                vectorstore, loader,
                days_back=args.days,
                query=args.query,
                max_messages=args.max
            )

            console.print(f"\n[green]Synced {count} email chunks to knowledge base.[/green]")

        elif args.sync_source == "drive":
            from src.integrations.drive import get_drive_client

            console.print("\n[bold cyan]📁 Google Drive Sync[/bold cyan]\n")

            client = get_drive_client()

            if not client.is_available():
                console.print("[red]Google API libraries not installed.[/red]")
                console.print("Run: [cyan]pip install google-auth-oauthlib google-api-python-client[/cyan]")
                sys.exit(1)

            if not client.is_configured():
                console.print("[red]Drive not configured.[/red]")
                console.print(f"Please save your OAuth credentials to: [cyan]~/.vault/credentials.json[/cyan]")
                sys.exit(1)

            console.print("Authenticating with Google Drive...")
            client.authenticate()
            quota = client.get_storage_quota()
            console.print(f"[green]Connected as:[/green] {quota['email']}")

            vectorstore = VectorStore()
            loader = DocumentLoader()

            count = client.sync_to_vault(
                vectorstore, loader,
                folder_id=args.folder,
                days_back=args.days,
                max_files=args.max
            )

            console.print(f"\n[green]Synced {count} files to knowledge base.[/green]")

        else:
            console.print("[yellow]Please specify a source: gmail or drive[/yellow]")
            console.print("  [cyan]python main.py sync gmail[/cyan]")
            console.print("  [cyan]python main.py sync drive[/cyan]")

    else:
        from src.cli import print_banner
        from rich.console import Console

        console = Console()
        print_banner()
        console.print("\n[bold]Quick Start:[/bold]")
        console.print("  1. Start Parallax:    [cyan]parallax run[/cyan]")
        console.print("  2. Ingest documents:  [cyan]python main.py ingest ~/Documents[/cyan]")
        console.print("  3. Start chatting:    [cyan]python main.py chat[/cyan]")
        console.print("  4. Or use web UI:     [cyan]python main.py web[/cyan]")
        console.print("\n[dim]Use --help for more options[/dim]")


if __name__ == "__main__":
    main()
