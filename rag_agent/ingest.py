import asyncio
import argparse
from .engine import Engine

async def main(prepare_existing_index: bool = False):
    engine = Engine()
    if prepare_existing_index:
        print(await asyncio.to_thread(engine.prepare_existing_index))
    else:
        print(await engine.rebuild())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the offline Notion/PDF vector and BM25 indexes")
    parser.add_argument("--prepare-existing-index", action="store_true",
                        help="Prepare BM25 from an existing vector index without fetching Notion or PDFs")
    args = parser.parse_args()
    asyncio.run(main(args.prepare_existing_index))
