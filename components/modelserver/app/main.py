import asyncio
import signal
import process
from config import config

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
shutdownEvent = asyncio.Event()


def signalHandler(signum, frame):
    loop.call_soon_threadsafe(shutdownEvent.set)


async def main():
    process.start()
    await shutdownEvent.wait()
    process.stop()

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signalHandler)
    signal.signal(signal.SIGTERM, signalHandler)
    loop.run_until_complete(main())
