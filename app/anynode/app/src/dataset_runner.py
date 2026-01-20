import os
import json
import re
import asyncio
import datetime
import random
from typing import List, Dict, Any
from modal import App, Image, Volume, Secret
import requests
import pandas as pd
from datasets import load_dataset
from scrapingbee import ScrapingBeeClient
import kaggle
import textblob  # For sentiment metadata

# Constants for cosmic math
PI = 3.1415926535
GOLDEN_RATIO = (1 + 5**0.5) / 2  # ~1.618
FIB_START = [8, 13, 20]  # Pi-scaled, capped for efficiency
LISSAJOUS_RATIO_NORMAL = 4  # 4:3 cycling
LISSAJOUS_RATIO_FIB = 3
SYNTH_SAMPLES = int(os.environ.get("SYNTH_SAMPLES", 314))  # ~100 * PI
REPORT_TOKENS = int(200 * PI / GOLDEN_RATIO)  # ~628 tokens est.
DOMAINS = [
    "Productivity", "Time Management", "Communication", "Email Management",
    "Account Management", "Team Collaboration", "Project Management and Decision Making",
    "Automation and Social Media Marketing"
]
SCRAPE_URLS_PER_DOMAIN = 12  # Optimized down from 15
KAGGLE_SAMPLES_PER_DOMAIN = 200  # Optimized down from 250
SCRAPE_CONCURRENCY = 15  # Increased from 10

# Modal setup
app = App("dataset-runner")
volume = Volume.from_name("datasets-vol", create_if_missing=True)  # Fixed line
image = Image.debian_slim().pip_install(
    "requests==2.31.0", "pandas==2.0.3", "datasets==2.14.4", "pyarrow==15.0.2",
    "scrapingbee==2.0.1", "kaggle==1.5.16", "textblob==0.18.0.post0"
)

@app.function(
    image=image,
    volumes={"/datasets": volume},
    secrets=[Secret.from_name("scrapingbee-api-keys"), Secret.from_name("kaggle-api-key"), Secret.from_name("lm-studio")],
    timeout=3600,  # 1 hour
)
async def run_dataset_gen(day: int = 1):
    # Load secrets and env
    scrapingbee_keys = parse_scrapingbee_keys(os.environ.get("SCRAPINGBEE_API_KEYS", ""))
    kaggle.authenticate()  # Uses KAGGLE_USERNAME and KAGGLE_KEY from secret
    lm_studio_url = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1")

    # Gabriel's Horn: In-memory data funnel
    all_data: Dict[str, List[Dict[str, Any]]] = {domain: [] for domain in DOMAINS}

    # Lissajous cycling: Alternate modes in 4:3 ratio
    for cycle in range(LISSAJOUS_RATIO_NORMAL + LISSAJOUS_RATIO_FIB):
        mode = "normal" if cycle < LISSAJOUS_RATIO_NORMAL else "fibonacci"
        domain_batch = DOMAINS[(cycle * 3) % len(DOMAINS): (cycle * 3 + 3) % len(DOMAINS)]  # ~PI domains/batch

        # Scrape data (concurrent)
        scraped = await scrape_domains_async(domain_batch, scrapingbee_keys)

        # Synthetic gen with Fibonacci batching and nutation
        synthetic = gen_synth(domain_batch, lm_studio_url, mode)

        # Kaggle pull with caching
        kaggle_data = pull_kaggle(domain_batch)

        # In-memory combine and filter
        for domain in domain_batch:
            domain_data = scraped.get(domain, []) + synthetic.get(domain, []) + kaggle_data.get(domain, [])
            domain_data = filter_relevant(domain_data, domain)  # Optimized filter
            all_data[domain].extend(domain_data)

    # Final write (JSONL)
    for domain, data in all_data.items():
        df = pd.DataFrame(data)
        path = f"/datasets/{domain}/train_day{day}.jsonl"
        df.to_json(path, orient="records", lines=True)
        volume.commit()  # Persist

    return {"status": "success", "samples": {d: len(all_data[d]) for d in DOMAINS}}

@app.function(
    image=image,
    secrets=[Secret.from_name("scrapingbee-api-keys"), Secret.from_name("kaggle-api-key"), Secret.from_name("lm-studio")],
    volumes={"/datasets": volume},
    timeout=3600,
    schedule=modal.Cron("0 14 * * 1-2"),
)
def scheduled_driver():
    day = datetime.date.today().day
    run_dataset_gen.spawn(day)

@app.local_entrypoint()
def main():
    for v in ("SCRAPINGBEE_API_KEYS", "KAGGLE_USERNAME", "KAGGLE_KEY", "LM_STUDIO_URL"):
        if not os.environ.get(v):
            raise ValueError(f"Set {v} in your PowerShell session or Modal secrets.")
    scheduled_driver.remote()
