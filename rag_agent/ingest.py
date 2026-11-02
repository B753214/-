import asyncio
from .engine import Engine

async def main():
    print(await Engine().rebuild())

if __name__ == "__main__":
    asyncio.run(main())
