"""Command-line version, handy for quick tests.

    python cli.py ingest sample_docs/*.md
    python cli.py ask "Compare the approaches used in these documents."
    python cli.py list
"""
import argparse

from docresearch.graph import ask
from docresearch.ingest import ingest_file
from docresearch.vectorstore import list_sources


def main() -> None:
    parser = argparse.ArgumentParser(description="Document Research Agent")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="Index one or more files")
    p_ingest.add_argument("files", nargs="+")

    p_ask = sub.add_parser("ask", help="Ask a question")
    p_ask.add_argument("question")

    sub.add_parser("list", help="Show indexed documents")
    args = parser.parse_args()

    if args.cmd == "ingest":
        for f in args.files:
            r = ingest_file(f)
            if r.error:
                print(f"x {r.file}: {r.error}")
            else:
                print(f"ok {r.file}: {r.pages} pages -> {r.chunks} chunks")
                for note in r.notes:
                    print(f"   note: {note}")

    elif args.cmd == "list":
        for s in list_sources() or ["(nothing indexed)"]:
            print(s)

    elif args.cmd == "ask":
        seen, final = 0, None
        for state in ask(args.question):
            final = state
            for line in state.get("trace", [])[seen:]:
                print(line)
            seen = len(state.get("trace", []))
        print("\n" + final["answer"])
        if final.get("sources"):
            print("\nSources:")
            for s in final["sources"]:
                print(f"  - {s['source']}, p.{s['page']}")


if __name__ == "__main__":
    main()
