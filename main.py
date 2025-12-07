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
