# connectors/discord/bot.py
import os, re, httpx
from dotenv import load_dotenv
import discord
from discord.ext import commands

load_dotenv()
TOKEN  =  os.getenv("DISCORD_BOT_TOKEN")
INGEST_URL  =  os.getenv("INGEST_URL","http://localhost:8080/ingest")
APPEND_URL  =  os.getenv("APPEND_URL","http://localhost:8080/append-answer")
QUESTION_LIKE  =  re.compile(r"(^|\s)(who|what|when|where|why|how|should|can|do|does|did|is|are|will)\b.*\?\s*$", re.I|re.S)

intents  =  discord.Intents.default()
intents.message_content  =  True
bot  =  commands.Bot(command_prefix = "!", intents = intents)

def is_question(msg: discord.Message) -> bool:
    if msg.author.bot: return False
    t  =  (msg.content or "").strip()
    return t.endswith("?") or bool(QUESTION_LIKE.search(t))

async def post_json(url, data):
    async with httpx.AsyncClient(timeout = 10) as c:
        r  =  await c.post(url, json = data); r.raise_for_status(); return r.json()

@bot.event
async def on_ready():
    print(f"Discord ready as {bot.user}")

@bot.event
async def on_message(msg: discord.Message):
    if is_question(msg):
        data  =  {
            "platform":"discord","question_id": str(msg.id),"question": msg.content,"url": msg.jump_url,
            "author_id": str(msg.author.id),"author_name": msg.author.display_name,
            "guild_id": str(msg.guild.id) if msg.guild else None,"channel_id": str(msg.channel.id),
            "thread_id": str(msg.thread.id) if isinstance(msg.channel, discord.Thread) else None,
            "timestamp": msg.created_at.isoformat()
        }
        try: await post_json(INGEST_URL, data); await msg.add_reaction("✅")
        except Exception as e: print("Ingest failed:", e)

    if msg.reference and msg.reference.message_id:
        qid  =  str(msg.reference.message_id)
        try: await post_json(f"{APPEND_URL}/{qid}", {"answer": msg.content})
        except Exception as e: print("Append failed:", e)
    await bot.process_commands(msg)

if __name__ == "__main__":
    bot.run(TOKEN)
