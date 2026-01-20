import tweepy
import instagrapi
import tiktok
import hvac
import os
from outfy import OutfyClient

class SocialMediaManager:
    def __init__(self):
        self.vault_client = hvac.Client(url=os.getenv('VAULT_URL'), token=os.getenv('VAULT_TOKEN'))
        self.twitter_creds = self.vault_client.secrets.kv.read_secret_version(path='twitter')['data']['data']
        self.instagram_creds = self.vault_client.secrets.kv.read_secret_version(path='instagram')['data']['data']
        self.tiktok_creds = self.vault_client.secrets.kv.read_secret_version(path='tiktok')['data']['data']
        self.outfy_creds = self.vault_client.secrets.kv.read_secret_version(path='outfy')['data']['data']

    def post_to_twitter(self, message, media=None):
        """Post to Twitter."""
        client = tweepy.Client(**self.twitter_creds)
        if media:
            media_id = client.upload_media(media)['media_id']
            return client.create_tweet(text=message, media_ids=[media_id])
        return client.create_tweet(text=message)

    def post_to_instagram(self, photo_path, caption):
        """Post to Instagram."""
        client = instagrapi.Client()
        client.login(**self.instagram_creds)
        return client.photo_upload(photo_path, caption)

    def post_to_tiktok(self, video_path, description):
        """Post to TikTok."""
        client = tiktok.Client(**self.tiktok_creds)
        return client.upload_video(video_path, description)

    def automate_social_posts(self, store_name, products):
        """Automate social media posts using Outfy."""
        outfy = OutfyClient(self.outfy_creds['api_key'])
        for product in products:
            outfy.create_post(
                product=product['title'],
                description=product['description'],
                image=product.get('image_url'),
                platforms=['twitter', 'instagram', 'tiktok']
            )
        return {'status': 'scheduled', 'store': store_name}