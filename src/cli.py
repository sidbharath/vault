"""Command-line interface for Vault."""

import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.markdown import Markdown

from .ingestion import DocumentLoader, get_supported_files
from .vectorstore import VectorStore
from .rag import RAGEngine

console = Console()


def print_banner():
    """Print the Vault banner."""
    banner = """
[bold cyan]╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   ██╗   ██╗ █████╗ ██╗   ██╗██╗  ████████╗               ║
║   ██║   ██║██╔══██╗██║   ██║██║  ╚══██╔══╝               ║
║   ██║   ██║███████║██║   ██║██║     ██║                  ║
║   ╚██╗ ██╔╝██╔══██║██║   ██║██║     ██║                  ║
║    ╚████╔╝ ██║  ██║╚██████╔╝███████╗██║                  ║
║     ╚═══╝  ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝                  ║
║                                                           ║
║   [white]Personal Knowledge Base powered by Parallax[/white]            ║
║   [dim]Your documents, your AI, 100% local & private[/dim]          ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝[/bold cyan]
"""
    console.print(banner)


def ingest_documents(path: str, vectorstore: VectorStore) -> int:
    """Ingest documents from a file or directory."""
    loader = DocumentLoader(chunk_size=500, chunk_overlap=50)
    path_obj = Path(path)

    if not path_obj.exists():
        console.print(f"[red]Error: Path not found: {path}[/red]")
        return 0

    chunks_added = 0

    if path_obj.is_file():
        files = [str(path_obj)]
    else:
        files = get_supported_files(str(path_obj))

    if not files:
        console.print("[yellow]No supported files found.[/yellow]")
        console.print(f"Supported formats: {', '.join(DocumentLoader.SUPPORTED_EXTENSIONS)}")
        return 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Processing documents...", total=len(files))

        for file_path in files:
            progress.update(task, description=f"Processing: {Path(file_path).name}")
            try:
                chunks = list(loader.process_file(file_path))
                added = vectorstore.add_chunks(chunks)
                chunks_added += added
            except Exception as e:
                console.print(f"[red]Error processing {file_path}: {e}[/red]")
            progress.advance(task)

    return chunks_added


def show_stats(vectorstore: VectorStore):
    """Display knowledge base statistics."""
    stats = vectorstore.get_stats()
    sources = vectorstore.list_sources()

    table = Table(title="Knowledge Base Stats")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Chunks", str(stats['total_chunks']))
    table.add_row("Documents", str(len(sources)))
    table.add_row("Storage", stats['persist_dir'])

    console.print(table)

    if sources:
        console.print("\n[bold]Indexed Documents:[/bold]")
        for source in sources:
            console.print(f"  - {Path(source).name}")


def interactive_chat(rag: RAGEngine):
    """Start an interactive chat session with conversation memory."""
    console.print("\n[bold green]Chat Mode[/bold green] - Ask questions about your documents")
    console.print("[dim]Commands: 'quit' | 'sources' | 'history' | 'clear' (reset conversation)[/dim]")

    # Check stats
    stats = rag.vectorstore.get_stats()
    console.print(f"[dim]Knowledge base: {stats['total_chunks']} chunks indexed[/dim]\n")

    # Check if Parallax is available
    if not rag.parallax.is_available():
        console.print(Panel(
            "[yellow]Parallax server not detected at localhost:3001\n\n"
            "Start Parallax with:[/yellow]\n"
            "[bold]parallax run[/bold]\n\n"
            "[dim]The knowledge base will still work for document search.[/dim]",
            title="Warning"
        ))
    else:
        console.print("[green]Parallax connected[/green] - Ready for conversation\n")

    while True:
        try:
            question = console.input("\n[bold cyan]You:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not question:
            continue

        # Handle commands
        if question.lower() in ('quit', 'exit', 'q'):
            console.print("[dim]Goodbye![/dim]")
            break

        if question.lower() == 'sources':
            sources = rag.vectorstore.list_sources()
            if sources:
                console.print("\n[bold]Indexed sources:[/bold]")
                for s in sources:
                    console.print(f"  - {s}")
            else:
                console.print("[yellow]No documents indexed yet.[/yellow]")
            continue

        if question.lower() == 'history':
            history = rag.get_history()
            if history:
                console.print("\n[bold]Conversation History:[/bold]")
                for i, msg in enumerate(history):
                    role = "[cyan]You[/cyan]" if msg.role == "user" else "[green]Vault[/green]"
                    content = msg.content[:150] + "..." if len(msg.content) > 150 else msg.content
                    console.print(f"  {role}: {content}")
            else:
                console.print("[dim]No conversation history yet.[/dim]")
            continue

        if question.lower() == 'clear':
            rag.clear_history()
            console.print("[green]Conversation history cleared. Starting fresh![/green]")
            continue

        if question.lower() == 'help':
            console.print(Panel(
                "[bold]Available Commands:[/bold]\n\n"
                "  [cyan]sources[/cyan]  - List indexed documents\n"
                "  [cyan]history[/cyan]  - Show conversation history\n"
                "  [cyan]clear[/cyan]    - Reset conversation (start fresh)\n"
                "  [cyan]quit[/cyan]     - Exit chat mode\n\n"
                "[dim]Tip: Ask follow-up questions! Vault remembers the conversation.[/dim]",
                title="Help"
            ))
            continue

        # Show retrieved sources
        sources = rag.get_sources_for_query(question)
        if sources:
            source_names = [Path(s['source']).name for s in sources[:3]]
            console.print(f"[dim]Searching: {', '.join(source_names)}...[/dim]")

        # Generate response
        console.print("\n[bold green]Vault:[/bold green] ", end="")

        if not rag.parallax.is_available():
            # Fallback: just show the context
            console.print("\n[yellow]Parallax not available. Showing retrieved context:[/yellow]\n")
            for i, source in enumerate(sources, 1):
                console.print(Panel(
                    source['content'][:500] + "..." if len(source['content']) > 500 else source['content'],
                    title=f"[{i}] {Path(source['source']).name}"
                ))
        else:
            try:
                for chunk in rag.query(question, stream=True):
                    console.print(chunk, end="")
                console.print()  # Newline after response
            except Exception as e:
                console.print(f"\n[red]Error: {e}[/red]")


def main():
    """Main entry point for the CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Vault - Personal Knowledge Base powered by Parallax"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("path", help="File or directory to ingest")

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Interactive chat")
    chat_parser.add_argument(
        "--url", default="http://localhost:3001",
        help="Parallax server URL"
    )

    # Search command
    search_parser = subparsers.add_parser("search", help="Search documents")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("-n", type=int, default=5, help="Number of results")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show statistics")

    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear all indexed documents")

    args = parser.parse_args()

    # Initialize vectorstore
    vectorstore = VectorStore()

    if args.command == "ingest":
        print_banner()
        console.print(f"\n[bold]Ingesting documents from:[/bold] {args.path}\n")
        count = ingest_documents(args.path, vectorstore)
        console.print(f"\n[green]Successfully indexed {count} chunks[/green]")
        show_stats(vectorstore)

    elif args.command == "chat":
        print_banner()
        rag = RAGEngine(vectorstore, parallax_url=args.url)
        interactive_chat(rag)

    elif args.command == "search":
        results = vectorstore.search(args.query, n_results=args.n)
        if results:
            console.print(f"\n[bold]Top {len(results)} results for:[/bold] {args.query}\n")
            for i, result in enumerate(results, 1):
                source = Path(result['source']).name
                distance = result.get('distance', 'N/A')
                console.print(Panel(
                    result['content'][:400] + "..." if len(result['content']) > 400 else result['content'],
                    title=f"[{i}] {source} (distance: {distance:.4f})" if isinstance(distance, float) else f"[{i}] {source}"
                ))
        else:
            console.print("[yellow]No results found.[/yellow]")

    elif args.command == "stats":
        print_banner()
        show_stats(vectorstore)

    elif args.command == "clear":
        if console.input("[red]Are you sure you want to clear all data? (yes/no): [/red]").lower() == "yes":
            vectorstore.clear()
            console.print("[green]Knowledge base cleared.[/green]")
        else:
            console.print("[dim]Cancelled.[/dim]")

    else:
        print_banner()
        console.print("\n[bold]Quick Start:[/bold]")
        console.print("  1. Ingest documents:  [cyan]vault ingest ~/Documents[/cyan]")
        console.print("  2. Start chatting:    [cyan]vault chat[/cyan]")
        console.print("\n[dim]Use --help for more options[/dim]")


if __name__ == "__main__":
    main()
