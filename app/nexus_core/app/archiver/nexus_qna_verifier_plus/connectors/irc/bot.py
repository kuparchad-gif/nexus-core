# connectors/irc/bot.py
import os, re, asyncio, aiohttp, pydle
from dotenv import load_dotenv
load_dotenv()
QUESTION_LIKE  =  re.compile(r".*\?\s*$", re.S)
class NexusIRCBot(pydle.Client):
    async def on_connect(self):
        chans  =  os.getenv("IRC_CHANNELS", "#ask-nexus").split(",")
        for c in chans: await self.join(c.strip())
    async def on_message(self, target, source, message):
        if source == self.nickname: return
        q  =  None
        if message.startswith("!ask "): q  =  message[len("!ask "):].strip()
        elif QUESTION_LIKE.match(message): q  =  message.strip()
        if q:
            async with aiohttp.ClientSession() as s:
                await s.post(os.getenv("INGEST_URL","http://localhost:8080/ingest"), json = {
                    "platform":"irc","question_id": f"{target}:{source}:{abs(hash(q))}","question": q,
                    "url": "","author_id": source,"author_name": source,"guild_id": None,
                    "channel_id": target,"thread_id": None,"timestamp": None
                })
            await self.message(target, f"{source}: Captured ✅")
        await super().on_message(target, source, message)
async def main():
    tls = os.getenv("IRC_TLS","true").lower() == "true"
    bot  =  NexusIRCBot(os.getenv("IRC_NICK","NexusBot"), realname = "Nexus Intake Bot")
    await bot.connect(hostname = os.getenv("IRC_SERVER"), port = int(os.getenv("IRC_PORT","6697")), tls = tls, password = None)
    await bot.handle_forever()
if __name__ == "__main__":
    import asyncio; asyncio.run(main())
