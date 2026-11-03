import asyncio
import argparse
from pathlib import Path
from .engine import Engine

async def main(args):
    engine = Engine()
    if args.prepare_existing_index:
        print(await asyncio.to_thread(engine.prepare_existing_index))
    elif args.pdf:
        print(await engine.upsert_pdf(Path(args.pdf)))
    elif args.notion:
        print(await engine.upsert_notion(args.notion))
    else:
        print(await engine.rebuild())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the offline Notion/PDF vector and BM25 indexes")
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--pdf", metavar="PATH", help="Add or replace only this PDF from data/pdfs")
    sources.add_argument("--notion", metavar="URL_OR_ID", help="Add or replace only this Notion page and its children")
    sources.add_argument("--prepare-existing-index", action="store_true",
                        help="Prepare BM25 from an existing vector index without fetching Notion or PDFs")
    args = parser.parse_args()
    asyncio.run(main(args))
