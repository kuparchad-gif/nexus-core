# Metatron Validator

Single-file validator that:
- Validates claims from multiple public sources (search or URLs)
- Flags: Fact, Opinion, Scientifically Proven, Spiritual Applications, Metaphor
- Outputs a Metatron's Cube mapping (13 nodes)
- Scrubs emails/phones/handles/IPs/addresses from evidence ("virgin data")

## Quick start
```bash
pip install httpx bs4 readability-lxml ddgs transformers  # optional but recommended
python metatron_validator.py --query "Coffee causes dehydration" --max-results 6
python metatron_validator.py --query "Metatron's Cube encodes Platonic solids" --no-search --sources https://en.wikipedia.org/wiki/Metatron%27s_Cube
python metatron_validator.py --query "Cold plunges improve recovery" --save-json out.json
```
