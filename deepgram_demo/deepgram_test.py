import asyncio
from time import time

async def task(name, delay):
    print(f"Task {name} started (waiting for {delay} seconds)")
    await asyncio.sleep(delay)
    print(f"Task {name} completed")
    return f"Result of {name}"

async def main():
    start_time = time()
    
    results = await asyncio.gather(
        task("A", 3),
        task("B", 1),
        task("C", 2)
    )
    end_time = time()
    print(f"Total time: {end_time - start_time}")
    print(f"Results: {results}")

# Run the program
if __name__ == "__main__":
    asyncio.run(main())