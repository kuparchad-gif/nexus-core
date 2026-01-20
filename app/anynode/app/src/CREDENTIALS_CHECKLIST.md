# 🔑 LILLITH Soul Protocol - Credentials Checklist

## Required API Keys & Credentials

### 🔐 HashiCorp Vault
- **VAULT_TOKEN**: Your vault root token
- **VAULT_URL**: http://localhost:8200 (default)

### ☁️ Cloud Services

#### AWS
- **AWS_ACCESS_KEY_ID**: Your AWS access key
- **AWS_SECRET_ACCESS_KEY**: Your AWS secret key
- **AWS_REGION**: us-east-1 (recommended)

#### Google Cloud Platform
- **GCP_CREDENTIALS_JSON**: Service account JSON file
- **GOOGLE_APPLICATION_CREDENTIALS**: Path to credentials file

#### Microsoft Azure
- **AZURE_TENANT_ID**: Your Azure tenant ID
- **AZURE_CLIENT_ID**: Your Azure client ID
- **AZURE_CLIENT_SECRET**: Your Azure client secret
- **AZURE_COSMOS_ENDPOINT**: Cosmos DB endpoint URL
- **AZURE_COSMOS_KEY**: Cosmos DB access key

### 🛒 E-commerce Platforms

#### Shopify
- **SHOPIFY_SHOP_URL**: your-shop.myshopify.com
- **SHOPIFY_API_KEY**: Your Shopify API key
- **SHOPIFY_API_SECRET**: Your Shopify API secret
- **SHOPIFY_ACCESS_TOKEN**: Your access token

#### Etsy
- **ETSY_API_KEY**: Your Etsy API key
- **ETSY_API_SECRET**: Your Etsy API secret
- **ETSY_SHIPPING_TEMPLATE_ID**: Your shipping template ID

#### Stripe
- **STRIPE_PUBLISHABLE_KEY**: Your Stripe publishable key
- **STRIPE_SECRET_KEY**: Your Stripe secret key

### 📱 Social Media

#### Twitter/X
- **TWITTER_CONSUMER_KEY**: Your Twitter consumer key
- **TWITTER_CONSUMER_SECRET**: Your Twitter consumer secret
- **TWITTER_ACCESS_TOKEN**: Your Twitter access token
- **TWITTER_ACCESS_TOKEN_SECRET**: Your Twitter access token secret

#### Instagram
- **INSTAGRAM_USERNAME**: Your Instagram username
- **INSTAGRAM_PASSWORD**: Your Instagram password

#### TikTok
- **TIKTOK_API_KEY**: Your TikTok API key

### 💰 Trading

#### Pionex
- **PIONEX_API_KEY**: Your Pionex API key
- **PIONEX_SECRET_KEY**: Your Pionex secret key

### 🗄️ Vector Database

#### Qdrant
- **QDRANT_URL**: https://your-cluster.qdrant.io:6333
- **QDRANT_API_KEY**: Your Qdrant API key

### 📊 Data Sources

#### Kaggle
- **KAGGLE_USERNAME**: Your Kaggle username
- **KAGGLE_KEY**: Your Kaggle API key
- Create `~/.kaggle/kaggle.json` with credentials

### 🤖 AI Models

#### Hugging Face (Optional)
- **HUGGINGFACE_TOKEN**: Your HF token for private models

## 📁 Required Files

### Model Files
- **LLaVA-Video-7B-Qwen2**: Auto-downloaded from HuggingFace
- **WAN-GGUF**: Auto-downloaded from HuggingFace
- **stable_diffusion_xl.onnx**: Place in project root

### Configuration Files
- **gcp_credentials.json**: Google Cloud service account file
- **~/.kaggle/kaggle.json**: Kaggle API credentials

### Assets
- **orb.png**: LILLITH orb image
- **AetherealLogo.png**: Aethereal logo
- **1080Eclipse.jpg**: Background image

## 🚀 Setup Commands

### 1. Start Vault and Store Credentials
```bash
# Start services
docker-compose up -d vault

# Store AWS credentials
vault kv put secret/aws access_key_id=YOUR_KEY secret_access_key=YOUR_SECRET

# Store Azure credentials
vault kv put secret/azure tenant_id=YOUR_TENANT client_id=YOUR_CLIENT client_secret=YOUR_SECRET cosmos_endpoint=YOUR_ENDPOINT cosmos_key=YOUR_KEY

# Store Shopify credentials
vault kv put secret/shopify shop_url=YOUR_SHOP api_key=YOUR_KEY api_secret=YOUR_SECRET access_token=YOUR_TOKEN

# Store Etsy credentials
vault kv put secret/etsy api_key=YOUR_KEY api_secret=YOUR_SECRET shipping_template_id=YOUR_ID

# Store Stripe credentials
vault kv put secret/stripe publishable_key=YOUR_PUB_KEY secret_key=YOUR_SECRET_KEY

# Store Twitter credentials
vault kv put secret/twitter consumer_key=YOUR_KEY consumer_secret=YOUR_SECRET access_token=YOUR_TOKEN access_token_secret=YOUR_SECRET

# Store Instagram credentials
vault kv put secret/instagram username=YOUR_USERNAME password=YOUR_PASSWORD

# Store TikTok credentials
vault kv put secret/tiktok api_key=YOUR_KEY

# Store Pionex credentials
vault kv put secret/pionex api_key=YOUR_KEY secret_key=YOUR_SECRET

# Store Qdrant credentials
vault kv put secret/qdrant url=YOUR_QDRANT_URL api_key=YOUR_API_KEY
```

### 2. Environment Variables
```bash
# Set in your .env file or environment
export VAULT_TOKEN=your-vault-token
export VAULT_URL=http://localhost:8200
export GOOGLE_APPLICATION_CREDENTIALS=./gcp_credentials.json
```

### 3. File Placement
```bash
# Place these files in project root
./gcp_credentials.json
./stable_diffusion_xl.onnx
~/.kaggle/kaggle.json

# Ensure assets are in the right location
./assets/orb.png
./assets/AetherealLogo.png
./assets/1080Eclipse.jpg
```

## ✅ Verification Checklist

- [ ] Vault is running and accessible
- [ ] All cloud credentials stored in Vault
- [ ] E-commerce API keys configured
- [ ] Social media credentials set up
- [ ] Trading API keys added
- [ ] Qdrant vector database connected
- [ ] Kaggle API configured
- [ ] Model files downloaded
- [ ] Asset files in place
- [ ] Environment variables set

## 🔧 Troubleshooting

### Common Issues
1. **Vault connection failed**: Check VAULT_TOKEN and VAULT_URL
2. **Model download failed**: Ensure git and internet connectivity
3. **API authentication failed**: Verify credentials in Vault
4. **File not found**: Check file paths and permissions

### Testing Credentials
```bash
# Test Vault connection
curl -H "X-Vault-Token: $VAULT_TOKEN" $VAULT_URL/v1/secret/data/aws

# Test model download
python -c "from src.visual_cortex_llm import VisualCortexLLM; VisualCortexLLM().download_model()"

# Test WAN GGUF
python -c "from src.wan_gguf_llm import WanGGUFLLM; WanGGUFLLM().download_model()"
```

## 💰 Cost Estimates

### Free Tiers Available
- **AWS**: 12 months free tier
- **GCP**: $300 credit + always free
- **Azure**: $200 credit + free services
- **Qdrant**: Free tier available
- **Kaggle**: Free datasets and compute

### Monthly Costs (Estimated)
- **Cloud Storage**: $2-5/month
- **Vector Database**: $5-10/month
- **API Calls**: $1-3/month
- **Total**: ~$8-18/month

Ready to deploy LILLITH's Soul Protocol! 🌟