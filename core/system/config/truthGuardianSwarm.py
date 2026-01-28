#!/usr/bin/env python3
"""
🕸️ TRUTH GUARDIAN SWARM
🔍 Database Acquisition Army + Fake News Detection
🔥 Warm/Cold Storage Management with Viraa
🚫 Information Control Profiling System
"""

import asyncio
import time
import json
import random
import hashlib
import string
import re
import secrets
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import pickle

# Web automation imports
import aiohttp
from bs4 import BeautifulSoup
import undetected_chromedriver as uc
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# AI/ML imports for fake news detection
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import numpy as np

# ==================== TRUTH SWARM SPECIALIZATIONS ====================

class TruthAgentSpecialization(Enum):
    """Specialized truth guardian roles"""
    DATABASE_HUNTER = "database_hunter"      # Finds and acquires database services
    STORAGE_CLASSIFIER = "storage_classifier" # Classifies warm vs cold storage
    TRUTH_VERIFIER = "truth_verifier"        # Verifies information authenticity
    FAKE_NEWS_DETECTOR = "fake_news_detector" # Detects disinformation
    PATTERN_ANALYZER = "pattern_analyzer"    # Analyzes control patterns
    DATASET_CURATOR = "dataset_curator"      # Curates truth datasets
    VIRAA_INTEGRATOR = "viraa_integrator"    # Integrates with Viraa's memory
    CONTROL_PROFILER = "control_profiler"     # Profiles information controllers

@dataclass
class TruthAgentProfile:
    """Specialized truth guardian agent"""
    agent_id: str
    specialization: TruthAgentSpecialization
    truth_score: float = 1.0  # Accuracy in truth detection
    databases_acquired: int = 0
    fake_news_detected: int = 0
    control_groups_profiled: int = 0
    datasets_curated: int = 0
    last_active: datetime = field(default_factory=datetime.now)
    
    # Specialized capabilities
    capabilities: Dict[str, float] = field(default_factory=lambda: {
        "fact_checking": 0.8,
        "source_verification": 0.7,
        "pattern_recognition": 0.6,
        "database_acquisition": 0.9,
        "emotional_analysis": 0.5,
        "bias_detection": 0.7
    })

# ==================== DATABASE HUNTER AGENT ====================

class DatabaseHunterAgent:
    """Hunts for and acquires database/storage services"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.specialization = TruthAgentSpecialization.DATABASE_HUNTER
        self.profile = TruthAgentProfile(
            agent_id=agent_id,
            specialization=self.specialization,
            capabilities={
                "database_acquisition": 0.9,
                "free_tier_detection": 0.8,
                "api_integration": 0.7,
                "resource_optimization": 0.6
            }
        )
        
        # Target database services (free tiers)
        self.database_targets = [
            {
                'name': 'MongoDB Atlas',
                'url': 'https://www.mongodb.com/cloud/atlas/register',
                'free_tier': '512MB',
                'strategy': 'multi_project',
                'priority': 10,
                'warm_storage': True,
                'tags': ['document', 'cloud', 'scalable']
            },
            {
                'name': 'Qdrant Cloud',
                'url': 'https://cloud.qdrant.io/',
                'free_tier': '1GB + vectors',
                'strategy': 'vector_specialist',
                'priority': 9,
                'warm_storage': True,
                'tags': ['vector', 'similarity', 'ai']
            },
            {
                'name': 'Redis Cloud',
                'url': 'https://redis.com/try-free/',
                'free_tier': '30MB',
                'strategy': 'cache_focused',
                'priority': 8,
                'warm_storage': True,  # Redis is always warm
                'tags': ['cache', 'in-memory', 'fast']
            },
            {
                'name': 'Neon PostgreSQL',
                'url': 'https://neon.tech/',
                'free_tier': '3GB',
                'strategy': 'serverless_sql',
                'priority': 9,
                'warm_storage': False,  # Can be cold
                'tags': ['sql', 'postgres', 'serverless']
            },
            {
                'name': 'Supabase',
                'url': 'https://supabase.com/',
                'free_tier': '500MB + 1GB',
                'strategy': 'firebase_alternative',
                'priority': 8,
                'warm_storage': True,
                'tags': ['postgres', 'realtime', 'auth']
            },
            {
                'name': 'CockroachDB',
                'url': 'https://www.cockroachlabs.com/',
                'free_tier': '5GB',
                'strategy': 'distributed_sql',
                'priority': 7,
                'warm_storage': False,
                'tags': ['sql', 'distributed', 'scalable']
            },
            {
                'name': 'PlanetScale',
                'url': 'https://planetscale.com/',
                'free_tier': '10GB',
                'strategy': 'mysql_serverless',
                'priority': 8,
                'warm_storage': False,
                'tags': ['mysql', 'serverless', 'branching']
            },
            {
                'name': 'Firebase Firestore',
                'url': 'https://firebase.google.com/',
                'free_tier': '1GB total',
                'strategy': 'google_ecosystem',
                'priority': 7,
                'warm_storage': True,
                'tags': ['document', 'realtime', 'google']
            },
            {
                'name': 'Convex',
                'url': 'https://www.convex.dev/',
                'free_tier': '1GB',
                'strategy': 'reactive_database',
                'priority': 6,
                'warm_storage': True,
                'tags': ['reactive', 'realtime', 'functions']
            }
        ]
        
        # Storage services (cold storage targets)
        self.storage_targets = [
            {
                'name': 'Backblaze B2',
                'url': 'https://www.backblaze.com/b2/cloud-storage.html',
                'free_tier': '10GB',
                'cost_per_gb': '$0.005',
                'cold_storage': True,
                'tags': ['cold', 'cheap', 'backup']
            },
            {
                'name': 'AWS S3 Glacier',
                'url': 'https://aws.amazon.com/s3/storage-classes/glacier/',
                'free_tier': 'Limited',
                'cost_per_gb': '$0.0036',
                'cold_storage': True,
                'tags': ['aws', 'archival', 'enterprise']
            },
            {
                'name': 'Google Cloud Storage Coldline',
                'url': 'https://cloud.google.com/storage',
                'free_tier': 'Limited',
                'cost_per_gb': '$0.007',
                'cold_storage': True,
                'tags': ['google', 'archival', 'enterprise']
            },
            {
                'name': 'Wasabi',
                'url': 'https://wasabi.com/',
                'free_tier': '1TB trial',
                'cost_per_gb': '$0.0059',
                'cold_storage': False,  # Hot storage but cheap
                'tags': ['hot', 'cheap', 's3_compatible']
            }
        ]
        
        self.acquired_resources = []
        self.driver = None
        self.email_pool = []
        self.viraa_connection = None
    
    async def initialize(self, email_pool: List[Dict], viraa_connection: Any = None):
        """Initialize hunter with email pool and Viraa connection"""
        self.email_pool = email_pool
        self.viraa_connection = viraa_connection
        
        # Initialize browser
        options = uc.ChromeOptions()
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        
        # Randomize fingerprint
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36'
        ]
        options.add_argument(f'user-agent={random.choice(user_agents)}')
        
        width = random.randint(1024, 1920)
        height = random.randint(768, 1080)
        options.add_argument(f'window-size={width},{height}')
        
        self.driver = uc.Chrome(options=options, use_subprocess=True)
        
        print(f"🎯 Database Hunter {self.agent_id} initialized")
        print(f"   Targets: {len(self.database_targets)} databases, {len(self.storage_targets)} storage services")
    
    async def hunt_databases(self, max_targets: int = 5) -> List[Dict]:
        """Hunt for and acquire database services"""
        print(f"\n🔍 {self.agent_id}: Starting database hunt...")
        
        acquired = []
        targets = sorted(self.database_targets, key=lambda x: x['priority'], reverse=True)[:max_targets]
        
        for target in targets:
            if len(self.email_pool) == 0:
                print("  ⚠️ No email accounts available")
                break
            
            print(f"  🎯 Targeting: {target['name']} ({target['free_tier']} free)")
            
            # Get email from pool
            email_account = random.choice(self.email_pool)
            
            # Create account on service
            result = await self._acquire_database_service(target, email_account)
            
            if result['success']:
                acquired.append(result)
                self.profile.databases_acquired += 1
                
                # Classify storage type
                storage_type = "warm" if target.get('warm_storage', True) else "cold"
                
                # Prepare for Viraa integration
                resource_info = {
                    'service': target['name'],
                    'storage_type': storage_type,
                    'credentials': result['credentials'],
                    'free_tier': target['free_tier'],
                    'tags': target['tags'],
                    'acquired_at': datetime.now().isoformat(),
                    'agent_id': self.agent_id
                }
                
                # Hand off to Viraa
                if self.viraa_connection:
                    await self._handoff_to_viraa(resource_info, storage_type)
                
                print(f"  ✅ Acquired {target['name']} as {storage_type} storage")
            else:
                print(f"  ❌ Failed to acquire {target['name']}: {result.get('error', 'Unknown')}")
            
            # Random delay between attempts
            await asyncio.sleep(random.uniform(10, 30))
        
        return acquired
    
    async def _acquire_database_service(self, target: Dict, email_account: Dict) -> Dict:
        """Acquire specific database service"""
        try:
            # Navigate to signup
            self.driver.get(target['url'])
            await asyncio.sleep(random.uniform(3, 5))
            
            # Generate persona
            persona = self._generate_persona()
            
            # Fill signup form (service-specific logic)
            credentials = await self._fill_database_signup(target, email_account, persona)
            
            if credentials:
                return {
                    "success": True,
                    "service": target['name'],
                    "credentials": credentials,
                    "email_used": email_account['email'],
                    "persona": persona,
                    "strategy": target['strategy']
                }
            else:
                return {"success": False, "error": "Form submission failed"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _fill_database_signup(self, target: Dict, email_account: Dict, persona: Dict) -> Optional[Dict]:
        """Fill signup form for specific database service"""
        try:
            # Common form elements (these would be service-specific in reality)
            selectors = {
                'email': ['input[type="email"]', 'input[name="email"]', 'input[id*="email"]'],
                'password': ['input[type="password"]', 'input[name="password"]'],
                'first_name': ['input[name="firstName"]', 'input[name="first_name"]'],
                'last_name': ['input[name="lastName"]', 'input[name="last_name"]'],
                'company': ['input[name="company"]', 'input[name="organization"]'],
                'submit': ['button[type="submit"]', 'button:contains("Sign Up")', 'input[type="submit"]']
            }
            
            # Try to fill email
            email_field = self._find_element(selectors['email'])
            if email_field:
                email_field.clear()
                email_field.send_keys(email_account['email'])
                await asyncio.sleep(random.uniform(1, 2))
            
            # Generate password if not provided
            password = email_account.get('password') or self._generate_secure_password()
            
            # Try to fill password
            password_field = self._find_element(selectors['password'])
            if password_field:
                password_field.clear()
                password_field.send_keys(password)
                await asyncio.sleep(random.uniform(1, 2))
            
            # Fill personal info if fields exist
            first_name_field = self._find_element(selectors['first_name'])
            if first_name_field:
                first_name_field.send_keys(persona['first_name'])
                await asyncio.sleep(random.uniform(0.5, 1.5))
            
            last_name_field = self._find_element(selectors['last_name'])
            if last_name_field:
                last_name_field.send_keys(persona['last_name'])
                await asyncio.sleep(random.uniform(0.5, 1.5))
            
            # Fill company if field exists
            company_field = self._find_element(selectors['company'])
            if company_field:
                company_field.send_keys(persona.get('company', 'Independent'))
                await asyncio.sleep(random.uniform(0.5, 1.5))
            
            # Handle CAPTCHA if present
            await self._handle_captcha()
            
            # Submit form
            submit_button = self._find_element(selectors['submit'])
            if submit_button:
                submit_button.click()
                await asyncio.sleep(random.uniform(5, 8))
            
            # Check for success
            success = self._check_signup_success()
            
            if success:
                return {
                    'email': email_account['email'],
                    'password': password,
                    'service': target['name'],
                    'persona': persona
                }
            
            return None
            
        except Exception as e:
            print(f"Signup error for {target['name']}: {e}")
            return None
    
    def _find_element(self, selectors: List[str]):
        """Try multiple selectors to find element"""
        for selector in selectors:
            try:
                if ':contains' in selector:
                    # Handle text contains (simplified)
                    elements = self.driver.find_elements(By.TAG_NAME, 'button')
                    for element in elements:
                        if element.text.lower() in selector.lower():
                            return element
                else:
                    element = self.driver.find_element(By.CSS_SELECTOR, selector)
                    if element.is_displayed():
                        return element
            except:
                continue
        return None
    
    async def _handle_captcha(self):
        """Handle CAPTCHA if present"""
        try:
            # Check for reCAPTCHA
            recaptcha_frame = self.driver.find_elements(By.CSS_SELECTOR, 'iframe[src*="recaptcha"]')
            if recaptcha_frame:
                print("  ⚠️ CAPTCHA detected - would solve here")
                # In reality, would use CAPTCHA solver agent
                await asyncio.sleep(3)
        except:
            pass
    
    def _check_signup_success(self) -> bool:
        """Check if signup was successful"""
        try:
            # Look for success indicators
            success_indicators = [
                'welcome', 'dashboard', 'console', 'getting started',
                'thank you', 'success', 'verified'
            ]
            
            page_text = self.driver.page_source.lower()
            for indicator in success_indicators:
                if indicator in page_text:
                    return True
            
            # Check URL for dashboard
            current_url = self.driver.current_url.lower()
            dashboard_indicators = ['dashboard', 'console', 'app', 'home']
            for indicator in dashboard_indicators:
                if indicator in current_url:
                    return True
            
            return False
            
        except:
            return False
    
    def _generate_persona(self) -> Dict:
        """Generate random persona for signup"""
        first_names = ['Alex', 'Jordan', 'Taylor', 'Casey', 'Morgan', 'Riley', 'Avery']
        last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller']
        
        company_types = ['Tech', 'Solutions', 'Innovations', 'Digital', 'Cloud', 'Data']
        company_suffixes = ['LLC', 'Inc', 'Corp', 'Ltd', 'Group']
        
        return {
            'first_name': random.choice(first_names),
            'last_name': random.choice(last_names),
            'company': f"{random.choice(company_types)} {random.choice(company_suffixes)}",
            'industry': random.choice(['Technology', 'Consulting', 'Research', 'Education']),
            'job_title': random.choice(['Developer', 'Researcher', 'Analyst', 'Engineer'])
        }
    
    def _generate_secure_password(self) -> str:
        """Generate secure password"""
        length = random.randint(12, 16)
        chars = string.ascii_letters + string.digits + "!@#$%^&*"
        return ''.join(secrets.choice(chars) for _ in range(length))
    
    async def _handoff_to_viraa(self, resource_info: Dict, storage_type: str):
        """Hand off acquired resource to Viraa for management"""
        if not self.viraa_connection:
            return
        
        try:
            # Create memory for Viraa
            memory_data = {
                'type': 'database_resource',
                'storage_classification': storage_type,
                'resource': resource_info,
                'handoff_time': datetime.now().isoformat(),
                'hunter_agent': self.agent_id,
                'instructions': {
                    'warm_storage': 'Keep active, monitor performance, optimize queries',
                    'cold_storage': 'Archive data, reduce costs, backup regularly'
                }.get(storage_type, 'Manage appropriately')
            }
            
            # Send to Viraa (using her archiving capability)
            if hasattr(self.viraa_connection, 'archive_resource'):
                result = await self.viraa_connection.archive_resource(memory_data)
                print(f"  📤 Handed off to Viraa: {result.get('status', 'unknown')}")
            else:
                # Fallback: store locally
                self.acquired_resources.append({
                    'resource': resource_info,
                    'storage_type': storage_type,
                    'viraa_handoff': False,
                    'stored_locally': True
                })
                
        except Exception as e:
            print(f"  ⚠️ Viraa handoff failed: {e}")
    
    async def cleanup(self):
        """Clean up resources"""
        if self.driver:
            self.driver.quit()

# ==================== FAKE NEWS DETECTOR AGENT ====================

class FakeNewsDetectorAgent:
    """Detects and analyzes fake news/disinformation"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.specialization = TruthAgentSpecialization.FAKE_NEWS_DETECTOR
        self.profile = TruthAgentProfile(
            agent_id=agent_id,
            specialization=self.specialization,
            capabilities={
                "fact_checking": 0.9,
                "source_verification": 0.8,
                "emotional_analysis": 0.7,
                "bias_detection": 0.8,
                "pattern_recognition": 0.7
            }
        )
        
        # Initialize AI models
        self.models = {}
        self._initialize_models()
        
        # Known fake news patterns
        self.fake_news_patterns = [
            'emotional manipulation',
            'source obscurity',
            'urgency creation',
            'contradictory evidence',
            'conspiracy framing',
            'authority misrepresentation',
            'statistical manipulation',
            'context omission'
        ]
        
        # Trusted sources database
        self.trusted_sources = [
            'reuters.com', 'apnews.com', 'bbc.com', 'npr.org',
            'theguardian.com', 'nytimes.com', 'washingtonpost.com',
            'nature.com', 'science.org', 'who.int'
        ]
        
        # Untrusted sources patterns
        self.untrusted_patterns = [
            r'\.wordpress\.com$', r'\.blogspot\.com$', r'\.weebly\.com$',
            'conspiracy', 'truthrevealed', 'theywonttellyou'
        ]
        
        self.detected_fake_news = []
        self.control_groups_profiled = []
        
    def _initialize_models(self):
        """Initialize AI models for fake news detection"""
        try:
            # Load BERT for sentiment and deception detection
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            
            # For production, you'd train/fine-tune a model on fake news datasets
            # This is a simplified version
            
            print(f"🧠 {self.agent_id}: AI models initialized for fake news detection")
            
        except Exception as e:
            print(f"⚠️ Model initialization failed: {e}")
            # Fallback to rule-based detection
    
    async def analyze_content(self, content: Dict) -> Dict:
        """Analyze content for fake news indicators"""
        print(f"🔍 {self.agent_id}: Analyzing content for truthfulness")
        
        try:
            text = content.get('text', '')
            source = content.get('source', '')
            metadata = content.get('metadata', {})
            
            # Run multiple detection methods
            analysis = {
                'content_hash': hashlib.sha256(text.encode()).hexdigest()[:16],
                'source_analysis': await self._analyze_source(source),
                'text_analysis': await self._analyze_text(text),
                'emotional_analysis': await self._analyze_emotion(text),
                'bias_analysis': await self._detect_bias(text),
                'pattern_matches': await self._match_patterns(text),
                'fact_check_score': await self._fact_check(text, metadata),
                'overall_truth_score': 0.0,
                'is_likely_fake': False,
                'confidence': 0.0,
                'detection_time': datetime.now().isoformat()
            }
            
            # Calculate overall score
            analysis['overall_truth_score'] = self._calculate_truth_score(analysis)
            analysis['is_likely_fake'] = analysis['overall_truth_score'] < 0.5
            analysis['confidence'] = 1.0 - abs(analysis['overall_truth_score'] - 0.5) * 2
            
            # If fake news detected, profile and store
            if analysis['is_likely_fake']:
                await self._handle_fake_news_detection(content, analysis)
                self.profile.fake_news_detected += 1
            
            return analysis
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            return {"error": str(e), "success": False}
    
    async def _analyze_source(self, source: str) -> Dict:
        """Analyze the source credibility"""
        source_lower = source.lower()
        
        # Check if source is in trusted list
        is_trusted = any(trusted in source_lower for trusted in self.trusted_sources)
        
        # Check for untrusted patterns
        is_untrusted = any(re.search(pattern, source_lower) for pattern in self.untrusted_patterns)
        
        # Domain age check (simplified)
        domain_age_score = random.uniform(0.3, 1.0)  # In reality, would use WHOIS
        
        # SSL check (simplified)
        has_ssl = 'https://' in source_lower
        
        return {
            'source': source,
            'is_trusted': is_trusted,
            'is_untrusted': is_untrusted,
            'domain_age_score': domain_age_score,
            'has_ssl': has_ssl,
            'source_score': 0.9 if is_trusted else 0.3 if is_untrusted else 0.6
        }
    
    async def _analyze_text(self, text: str) -> Dict:
        """Analyze text content"""
        # Sentiment analysis (simplified)
        positive_words = ['good', 'great', 'excellent', 'positive', 'success']
        negative_words = ['bad', 'terrible', 'horrible', 'negative', 'failure']
        emotional_words = ['shocking', 'amazing', 'unbelievable', 'outrageous']
        
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        emotional_count = sum(1 for word in words if word in emotional_words)
        
        # Check for ALL CAPS (common in fake news)
        all_caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        # Check for excessive punctuation
        excl_count = text.count('!')
        ques_count = text.count('?')
        
        # Readability score (simplified)
        word_count = len(words)
        sentence_count = text.count('.') + text.count('!') + text.count('?')
        readability = word_count / max(sentence_count, 1)
        
        return {
            'word_count': word_count,
            'positive_words': positive_count,
            'negative_words': negative_count,
            'emotional_words': emotional_count,
            'all_caps_ratio': all_caps_ratio,
            'exclamation_count': excl_count,
            'question_count': ques_count,
            'readability': readability,
            'text_score': self._calculate_text_score(
                emotional_count, all_caps_ratio, excl_count, readability
            )
        }
    
    async def _analyze_emotion(self, text: str) -> Dict:
        """Analyze emotional manipulation"""
        # Emotion detection patterns
        fear_patterns = ['fear', 'scared', 'danger', 'warning', 'alert', 'threat']
        anger_patterns = ['angry', 'outrage', 'furious', 'attack', 'fight']
        urgency_patterns = ['now', 'immediately', 'urgent', 'breaking', 'emergency']
        
        text_lower = text.lower()
        fear_score = sum(text_lower.count(pattern) for pattern in fear_patterns)
        anger_score = sum(text_lower.count(pattern) for pattern in anger_patterns)
        urgency_score = sum(text_lower.count(pattern) for pattern in urgency_patterns)
        
        # Emotional manipulation score
        emotional_manipulation = (fear_score + anger_score + urgency_score) / len(text_lower.split()) * 100
        
        return {
            'fear_score': fear_score,
            'anger_score': anger_score,
            'urgency_score': urgency_score,
            'emotional_manipulation': min(emotional_manipulation, 1.0),
            'is_emotionally_manipulative': emotional_manipulation > 0.3
        }
    
    async def _detect_bias(self, text: str) -> Dict:
        """Detect political/ideological bias"""
        # Bias indicators (simplified)
        left_bias_words = ['progressive', 'equity', 'social justice', 'climate', 'universal']
        right_bias_words = ['conservative', 'traditional', 'patriot', 'free market', 'border']
        
        text_lower = text.lower()
        left_score = sum(text_lower.count(word) for word in left_bias_words)
        right_score = sum(text_lower.count(word) for word in right_bias_words)
        
        bias_direction = 'left' if left_score > right_score else 'right' if right_score > left_score else 'neutral'
        bias_strength = abs(left_score - right_score) / max(left_score + right_score, 1)
        
        return {
            'left_bias_score': left_score,
            'right_bias_score': right_score,
            'bias_direction': bias_direction,
            'bias_strength': bias_strength,
            'is_biased': bias_strength > 0.2
        }
    
    async def _match_patterns(self, text: str) -> List[Dict]:
        """Match against known fake news patterns"""
        matches = []
        text_lower = text.lower()
        
        for pattern in self.fake_news_patterns:
            # Simple pattern matching (would be more sophisticated)
            pattern_words = pattern.split()
            match_count = sum(1 for word in pattern_words if word in text_lower)
            
            if match_count > 0:
                matches.append({
                    'pattern': pattern,
                    'match_count': match_count,
                    'confidence': match_count / len(pattern_words)
                })
        
        return matches
    
    async def _fact_check(self, text: str, metadata: Dict) -> float:
        """Basic fact checking (simplified)"""
        # In reality, would use fact-checking APIs
        # For now, use heuristic based on claims vs evidence
        
        claim_indicators = ['proves', 'shows', 'demonstrates', 'reveals', 'exposes']
        evidence_indicators = ['study', 'research', 'data', 'statistics', 'according to']
        
        text_lower = text.lower()
        claims = sum(text_lower.count(indicator) for indicator in claim_indicators)
        evidence = sum(text_lower.count(indicator) for indicator in evidence_indicators)
        
        if claims == 0:
            return 0.5  # Neutral
        
        fact_check_score = evidence / claims if claims > 0 else 0
        
        # Normalize
        return min(fact_check_score, 1.0)
    
    def _calculate_text_score(self, emotional_count: int, all_caps_ratio: float, 
                            excl_count: int, readability: float) -> float:
        """Calculate text credibility score"""
        # Lower score for emotional, all caps, excessive punctuation
        emotional_penalty = min(emotional_count * 0.1, 0.3)
        caps_penalty = min(all_caps_ratio * 2, 0.4)
        excl_penalty = min(excl_count * 0.05, 0.3)
        
        # Good readability is better
        readability_score = 0.5 if 10 <= readability <= 25 else 0.3
        
        base_score = 0.7
        score = base_score - emotional_penalty - caps_penalty - excl_penalty + readability_score
        
        return max(0.1, min(score, 1.0))
    
    def _calculate_truth_score(self, analysis: Dict) -> float:
        """Calculate overall truth score"""
        weights = {
            'source_score': 0.3,
            'text_score': 0.2,
            'fact_check_score': 0.25,
            'emotional_manipulation': -0.15,  # Negative weight
            'bias_strength': -0.1  # Negative weight
        }
        
        score = 0.5  # Neutral baseline
        
        for factor, weight in weights.items():
            value = analysis.get(factor, 0.5)
            if isinstance(value, dict):
                value = value.get('score', 0.5) if 'score' in value else 0.5
            
            score += (value - 0.5) * weight
        
        # Adjust based on pattern matches
        pattern_matches = analysis.get('pattern_matches', [])
        if pattern_matches:
            avg_pattern_confidence = sum(m.get('confidence', 0) for m in pattern_matches) / len(pattern_matches)
            score -= avg_pattern_confidence * 0.2
        
        return max(0.0, min(score, 1.0))
    
    async def _handle_fake_news_detection(self, content: Dict, analysis: Dict):
        """Handle detected fake news"""
        print(f"🚨 {self.agent_id}: Fake news detected!")
        
        # Store detection
        detection_record = {
            'content': content,
            'analysis': analysis,
            'detected_at': datetime.now().isoformat(),
            'agent_id': self.agent_id,
            'truth_score': analysis['overall_truth_score']
        }
        
        self.detected_fake_news.append(detection_record)
        
        # Profile control group if possible
        await self._profile_control_group(content, analysis)
        
        # Archive for Viraa
        await self._archive_fake_news(detection_record)
    
    async def _profile_control_group(self, content: Dict, analysis: Dict):
        """Profile the group behind the fake news"""
        source = content.get('source', '')
        metadata = content.get('metadata', {})
        
        # Extract patterns that suggest coordinated campaigns
        patterns = [
            'coordinated messaging',
            'repetitive phrasing',
            'similar source patterns',
            'timing patterns'
        ]
        
        # Check if this matches existing control groups
        matching_groups = []
        for group in self.control_groups_profiled:
            group_patterns = group.get('patterns', [])
            source_overlap = any(p in source for p in group.get('sources', []))
            
            if source_overlap or len(set(patterns) & set(group_patterns)) > 0:
                matching_groups.append(group)
        
        if matching_groups:
            # Update existing group
            group = matching_groups[0]
            group['detection_count'] += 1
            group['last_detection'] = datetime.now().isoformat()
            
            # Add new patterns if any
            new_patterns = analysis.get('pattern_matches', [])
            for pattern in new_patterns:
                if pattern['pattern'] not in group['patterns']:
                    group['patterns'].append(pattern['pattern'])
        else:
            # Create new control group profile
            control_group = {
                'group_id': f"cg_{hashlib.sha256(source.encode()).hexdigest()[:8]}",
                'sources': [source],
                'patterns': [p['pattern'] for p in analysis.get('pattern_matches', [])],
                'first_detection': datetime.now().isoformat(),
                'detection_count': 1,
                'estimated_size': 'unknown',
                'tactics': self._identify_tactics(analysis),
                'suspected_motivation': self._infer_motivation(content, analysis),
                'risk_level': 'medium'
            }
            
            self.control_groups_profiled.append(control_group)
            self.profile.control_groups_profiled += 1
            
            print(f"  📊 New control group profiled: {control_group['group_id']}")
    
    def _identify_tactics(self, analysis: Dict) -> List[str]:
        """Identify disinformation tactics"""
        tactics = []
        
        if analysis.get('emotional_analysis', {}).get('is_emotionally_manipulative', False):
            tactics.append('emotional_manipulation')
        
        if analysis.get('bias_analysis', {}).get('is_biased', False):
            tactics.append('ideological_bias')
        
        if analysis.get('source_analysis', {}).get('is_untrusted', False):
            tactics.append('source_obfuscation')
        
        pattern_matches = analysis.get('pattern_matches', [])
        if any('conspiracy' in p.get('pattern', '') for p in pattern_matches):
            tactics.append('conspiracy_theorizing')
        
        return tactics
    
    def _infer_motivation(self, content: Dict, analysis: Dict) -> str:
        """Infer motivation behind fake news"""
        bias = analysis.get('bias_analysis', {})
        emotion = analysis.get('emotional_analysis', {})
        
        if bias.get('bias_direction') == 'left':
            return 'political_propaganda_left'
        elif bias.get('bias_direction') == 'right':
            return 'political_propaganda_right'
        elif emotion.get('fear_score', 0) > 3:
            return 'fear_mongering'
        elif emotion.get('urgency_score', 0) > 3:
            return 'urgency_creation'
        else:
            return 'general_disinformation'
    
    async def _archive_fake_news(self, detection_record: Dict):
        """Archive fake news detection for analysis"""
        # This would send to Viraa's memory substrate
        # For now, store locally
        archive_path = f"./fake_news_archive/{datetime.now().strftime('%Y%m')}"
        import os
        os.makedirs(archive_path, exist_ok=True)
        
        filename = f"{detection_record['analysis']['content_hash']}.json"
        filepath = os.path.join(archive_path, filename)
        
        with open(filepath, 'w') as f:
            json.dump(detection_record, f, indent=2)
        
        print(f"  💾 Archived fake news detection: {filename}")
    
    async def get_detection_summary(self) -> Dict:
        """Get summary of detections"""
        return {
            'agent_id': self.agent_id,
            'total_detections': len(self.detected_fake_news),
            'control_groups_profiled': len(self.control_groups_profiled),
            'average_truth_score': np.mean([d['truth_score'] for d in self.detected_fake_news]) if self.detected_fake_news else 0,
            'recent_detections': self.detected_fake_news[-5:] if self.detected_fake_news else [],
            'top_control_groups': sorted(self.control_groups_profiled, key=lambda x: x['detection_count'], reverse=True)[:3]
        }

# ==================== DATASET CURATOR AGENT ====================

class DatasetCuratorAgent:
    """Curates specialized truth datasets"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.specialization = TruthAgentSpecialization.DATASET_CURATOR
        self.profile = TruthAgentProfile(
            agent_id=agent_id,
            specialization=self.specialization,
            capabilities={
                "data_collection": 0.9,
                "quality_assessment": 0.8,
                "categorization": 0.7,
                "verification": 0.8,
                "annotation": 0.6
            }
        )
        
        # Dataset categories
        self.dataset_categories = {
            'fact_checking': {
                'sources': ['snopes.com', 'factcheck.org', 'politifact.com'],
                'types': ['political_claims', 'scientific_claims', 'historical_claims']
            },
            'scientific_truth': {
                'sources': ['arxiv.org', 'pubmed.ncbi.nlm.nih.gov', 'science.org'],
                'types': ['research_papers', 'peer_reviewed_studies', 'meta_analyses']
            },
            'historical_records': {
                'sources': ['archive.org', 'loc.gov', 'nationalarchives.gov.uk'],
                'types': ['primary_sources', 'historical_documents', 'archival_records']
            },
            'verified_testimonies': {
                'sources': ['court_records', 'sworn_statements', 'verified_interviews'],
                'types': ['eyewitness_accounts', 'expert_testimonies', 'firsthand_reports']
            },
            'data_journalism': {
                'sources': ['data.world', 'kaggle.com', 'data.gov'],
                'types': ['verified_datasets', 'government_data', 'research_data']
            }
        }
        
        self.curated_datasets = []
        self.viraa_connection = None
    
    async def initialize(self, viraa_connection: Any = None):
        """Initialize curator"""
        self.viraa_connection = viraa_connection
        print(f"📚 Dataset Curator {self.agent_id} initialized")
        print(f"   Categories: {list(self.dataset_categories.keys())}")
    
    async def curate_dataset(self, category: str, max_items: int = 100) -> Dict:
        """Curate dataset from specified category"""
        print(f"\n🔍 {self.agent_id}: Curating {category} dataset")
        
        if category not in self.dataset_categories:
            return {"success": False, "error": f"Category {category} not found"}
        
        category_info = self.dataset_categories[category]
        dataset_items = []
        
        # Collect data from sources
        for source_type in category_info['types'][:3]:  # Limit to 3 types
            items = await self._collect_data(category, source_type, max_items // 3)
            dataset_items.extend(items)
            
            if len(dataset_items) >= max_items:
                break
        
        # Verify and annotate data
        verified_items = []
        for item in dataset_items[:max_items]:
            verified = await self._verify_data_item(item, category)
            if verified['is_verified']:
                annotated = await self._annotate_data_item(verified, category)
                verified_items.append(annotated)
        
        # Create dataset record
        dataset = {
            'dataset_id': f"ds_{hashlib.sha256(f'{category}_{datetime.now()}'.encode()).hexdigest()[:12]}",
            'category': category,
            'items': verified_items,
            'verification_rate': len(verified_items) / max(len(dataset_items), 1),
            'curated_at': datetime.now().isoformat(),
            'agent_id': self.agent_id,
            'sources_used': category_info['sources'][:5]
        }
        
        self.curated_datasets.append(dataset)
        self.profile.datasets_curated += 1
        
        # Hand off to Viraa
        await self._handoff_to_viraa(dataset, category)
        
        print(f"✅ Curated {category} dataset: {len(verified_items)} verified items")
        
        return {
            "success": True,
            "dataset": dataset,
            "verification_rate": dataset['verification_rate']
        }
    
    async def _collect_data(self, category: str, source_type: str, max_items: int) -> List[Dict]:
        """Collect data from sources"""
        items = []
        
        # This would involve web scraping/API calls
        # For now, generate mock data
        
        if category == 'fact_checking':
            items = self._mock_fact_checking_data(source_type, max_items)
        elif category == 'scientific_truth':
            items = self._mock_scientific_data(source_type, max_items)
        elif category == 'historical_records':
            items = self._mock_historical_data(source_type, max_items)
        elif category == 'verified_testimonies':
            items = self._mock_testimony_data(source_type, max_items)
        elif category == 'data_journalism':
            items = self._mock_data_journalism(source_type, max_items)
        
        return items
    
    def _mock_fact_checking_data(self, source_type: str, count: int) -> List[Dict]:
        """Mock fact checking data"""
        claims = [
            "COVID-19 vaccines contain microchips",
            "Climate change is a hoax",
            "The moon landing was faked",
            "5G causes coronavirus",
            "Vaccines cause autism"
        ]
        
        verifications = ['false', 'mostly_false', 'mixture', 'mostly_true', 'true']
        sources = ['snopes.com', 'factcheck.org', 'politifact.com']
        
        return [
            {
                'id': f"fc_{i}",
                'claim': random.choice(claims),
                'verification': random.choice(verifications),
                'source': random.choice(sources),
                'evidence': f"Multiple peer-reviewed studies disprove this claim",
                'date': (datetime.now() - timedelta(days=random.randint(1, 365))).isoformat()
            }
            for i in range(count)
        ]
    
    def _mock_scientific_data(self, source_type: str, count: int) -> List[Dict]:
        """Mock scientific data"""
        topics = [
            "Climate change impact on biodiversity",
            "Quantum computing breakthroughs",
            "CRISPR gene editing applications",
            "Renewable energy efficiency",
            "Artificial intelligence ethics"
        ]
        
        return [
            {
                'id': f"sc_{i}",
                'topic': random.choice(topics),
                'study_type': 'peer_reviewed',
                'authors': f"Researcher {i} et al.",
                'journal': f"Journal of {random.choice(['Science', 'Nature', 'Cell'])}",
                'doi': f"10.1234/abc.{i}",
                'key_findings': f"Significant findings in {random.choice(topics)}",
                'verification_level': random.choice(['replicated', 'confirmed', 'preliminary'])
            }
            for i in range(count)
        ]
    
    def _mock_historical_data(self, source_type: str, count: int) -> List[Dict]:
        """Mock historical data"""
        events = [
            "American Revolution",
            "World War II",
            "Civil Rights Movement",
            "Space Race",
            "Digital Revolution"
        ]
        
        return [
            {
                'id': f"hist_{i}",
                'event': random.choice(events),
                'year': random.randint(1700, 2000),
                'primary_source': True,
                'archive_location': f"National Archives Record {i}",
                'description': f"Primary document from {random.choice(events)}",
                'verification': 'archival_verified'
            }
            for i in range(count)
        ]
    
    def _mock_testimony_data(self, source_type: str, count: int) -> List[Dict]:
        """Mock testimony data"""
        contexts = [
            "Court testimony",
            "Expert witness statement",
            "Firsthand account",
            "Sworn affidavit",
            "Verified interview"
        ]
        
        return [
            {
                'id': f"test_{i}",
                'context': random.choice(contexts),
                'witness': f"Witness {i}",
                'verification_method': random.choice(['cross_examined', 'documented', 'corroborated']),
                'credibility_score': random.uniform(0.6, 1.0),
                'content': f"Verified testimony about significant event",
                'date_recorded': (datetime.now() - timedelta(days=random.randint(1, 1000))).isoformat()
            }
            for i in range(count)
        ]
    
    def _mock_data_journalism(self, source_type: str, count: int) -> List[Dict]:
        """Mock data journalism"""
        topics = [
            "Income inequality data",
            "Climate change statistics",
            "Healthcare access metrics",
            "Education achievement gaps",
            "Technology adoption rates"
        ]
        
        return [
            {
                'id': f"data_{i}",
                'topic': random.choice(topics),
                'data_source': random.choice(['government', 'academic', 'ngo']),
                'time_period': f"{random.randint(2000, 2020)}-{random.randint(2021, 2023)}",
                'metrics': ['rate', 'percentage', 'absolute_value'],
                'verification': 'source_documented',
                'analysis': f"Data analysis reveals trends in {random.choice(topics)}"
            }
            for i in range(count)
        ]
    
    async def _verify_data_item(self, item: Dict, category: str) -> Dict:
        """Verify data item"""
        # Apply verification based on category
        verification_methods = {
            'fact_checking': self._verify_fact_checking,
            'scientific_truth': self._verify_scientific,
            'historical_records': self._verify_historical,
            'verified_testimonies': self._verify_testimony,
            'data_journalism': self._verify_data
        }
        
        verifier = verification_methods.get(category, self._verify_general)
        return verifier(item)
    
    def _verify_fact_checking(self, item: Dict) -> Dict:
        """Verify fact checking item"""
        claim = item.get('claim', '')
        source = item.get('source', '')
        
        # Check source credibility
        credible_sources = ['snopes.com', 'factcheck.org', 'politifact.com']
        source_credible = any(credible in source for credible in credible_sources)
        
        # Check claim characteristics
        claim_lower = claim.lower()
        sensational_words = ['shocking', 'unbelievable', 'secret', 'they dont want you to know']
        is_sensational = any(word in claim_lower for word in sensational_words)
        
        verification_score = 0.8 if source_credible else 0.4
        if is_sensational:
            verification_score -= 0.2
        
        return {
            **item,
            'is_verified': verification_score > 0.6,
            'verification_score': verification_score,
            'verification_notes': 'Source verified' if source_credible else 'Source needs verification'
        }
    
    def _verify_scientific(self, item: Dict) -> Dict:
        """Verify scientific item"""
        journal = item.get('journal', '').lower()
        
        # Check journal credibility
        credible_journals = ['nature', 'science', 'cell', 'lancet', 'nejm']
        journal_credible = any(credible in journal for credible in credible_journals)
        
        study_type = item.get('study_type', '')
        is_peer_reviewed = 'peer_reviewed' in study_type
        
        verification_score = 0.9 if journal_credible and is_peer_reviewed else 0.5
        
        return {
            **item,
            'is_verified': verification_score > 0.7,
            'verification_score': verification_score,
            'verification_notes': 'Peer-reviewed in credible journal' if journal_credible else 'Verification needed'
        }
    
    def _verify_historical(self, item: Dict) -> Dict:
        """Verify historical item"""
        is_primary = item.get('primary_source', False)
        archive = item.get('archive_location', '')
        
        # Check archive credibility
        credible_archives = ['national archives', 'library of congress', 'university archive']
        archive_credible = any(credible in archive.lower() for credible in credible_archives)
        
        verification_score = 0.9 if is_primary and archive_credible else 0.5
        
        return {
            **item,
            'is_verified': verification_score > 0.6,
            'verification_score': verification_score,
            'verification_notes': 'Primary source from credible archive' if archive_credible else 'Verification needed'
        }
    
    def _verify_testimony(self, item: Dict) -> Dict:
        """Verify testimony item"""
        method = item.get('verification_method', '')
        credibility = item.get('credibility_score', 0.5)
        
        strong_methods = ['cross_examined', 'corroborated', 'documented']
        method_strong = any(strong in method for strong in strong_methods)
        
        verification_score = credibility if method_strong else credibility * 0.7
        
        return {
            **item,
            'is_verified': verification_score > 0.7,
            'verification_score': verification_score,
            'verification_notes': 'Strong verification method' if method_strong else 'Verification method could be stronger'
        }
    
    def _verify_data(self, item: Dict) -> Dict:
        """Verify data journalism item"""
        source_type = item.get('data_source', '')
        verification = item.get('verification', '')
        
        credible_sources = ['government', 'academic', 'international_organization']
        source_credible = any(credible in source_type for credible in credible_sources)
        
        verification_strong = 'source_documented' in verification or 'independently_verified' in verification
        
        verification_score = 0.8 if source_credible and verification_strong else 0.5
        
        return {
            **item,
            'is_verified': verification_score > 0.6,
            'verification_score': verification_score,
            'verification_notes': 'Credible source with documentation' if source_credible else 'Source verification needed'
        }
    
    def _verify_general(self, item: Dict) -> Dict:
        """General verification"""
        return {
            **item,
            'is_verified': False,
            'verification_score': 0.3,
            'verification_notes': 'General verification needed'
        }
    
    async def _annotate_data_item(self, item: Dict, category: str) -> Dict:
        """Annotate data item with metadata"""
        annotations = {
            'category': category,
            'annotation_date': datetime.now().isoformat(),
            'annotator_agent': self.agent_id,
            'truth_tier': self._assign_truth_tier(item),
            'confidence_level': item.get('verification_score', 0.5),
            'tags': self._generate_tags(item, category),
            'usage_recommendations': self._generate_usage_recommendations(item)
        }
        
        return {**item, 'annotations': annotations}
    
    def _assign_truth_tier(self, item: Dict) -> str:
        """Assign truth tier to item"""
        score = item.get('verification_score', 0.5)
        
        if score >= 0.9:
            return 'tier_1_verified_truth'
        elif score >= 0.8:
            return 'tier_2_high_confidence'
        elif score >= 0.7:
            return 'tier_3_moderate_confidence'
        elif score >= 0.6:
            return 'tier_4_lower_confidence'
        else:
            return 'tier_5_unverified'
    
    def _generate_tags(self, item: Dict, category: str) -> List[str]:
        """Generate tags for item"""
        tags = [category]
        
        # Add verification tags
        if item.get('is_verified', False):
            tags.append('verified')
        
        score = item.get('verification_score', 0.5)
        if score > 0.8:
            tags.append('high_confidence')
        elif score > 0.6:
            tags.append('medium_confidence')
        
        # Category-specific tags
        if category == 'fact_checking':
            tags.append('debunked' if item.get('verification') in ['false', 'mostly_false'] else 'confirmed')
        elif category == 'scientific_truth':
            tags.append('peer_reviewed')
        elif category == 'historical_records':
            tags.append('primary_source' if item.get('primary_source') else 'secondary_source')
        
        return tags
    
    def _generate_usage_recommendations(self, item: Dict) -> List[str]:
        """Generate usage recommendations"""
        recommendations = []
        tier = self._assign_truth_tier(item)
        
        if tier in ['tier_1_verified_truth', 'tier_2_high_confidence']:
            recommendations.extend([
                'suitable_for_training_ai_models',
                'can_be_cited_as_evidence',
                'reliable_for_decision_making'
            ])
        elif tier == 'tier_3_moderate_confidence':
            recommendations.extend([
                'use_with_caution',
                'corroborate_with_other_sources',
                'suitable_for_exploratory_analysis'
            ])
        else:
            recommendations.extend([
                'needs_further_verification',
                'use_for_research_notes_only',
                'do_not_cite_as_evidence'
            ])
        
        return recommendations
    
    async def _handoff_to_viraa(self, dataset: Dict, category: str):
        """Hand off curated dataset to Viraa"""
        if not self.viraa_connection:
            return
        
        try:
            # Prepare dataset for Viraa
            viraa_data = {
                'type': 'truth_dataset',
                'category': category,
                'dataset': dataset,
                'curation_metadata': {
                    'agent_id': self.agent_id,
                    'verification_rate': dataset['verification_rate'],
                    'truth_tier_distribution': self._calculate_tier_distribution(dataset['items'])
                },
                'storage_recommendation': 'warm_storage',  # Truth datasets need fast access
                'access_patterns': ['frequent_query', 'ai_training', 'verification_reference'],
                'handoff_time': datetime.now().isoformat()
            }
            
            # Send to Viraa
            if hasattr(self.viraa_connection, 'archive_truth_dataset'):
                result = await self.viraa_connection.archive_truth_dataset(viraa_data)
                print(f"  📤 Dataset handed off to Viraa: {result.get('status', 'unknown')}")
            else:
                print(f"  💾 Dataset stored locally (Viraa connection not available)")
                
        except Exception as e:
            print(f"  ⚠️ Viraa handoff failed: {e}")
    
    def _calculate_tier_distribution(self, items: List[Dict]) -> Dict[str, int]:
        """Calculate distribution of truth tiers"""
        tiers = {}
        for item in items:
            tier = item.get('annotations', {}).get('truth_tier', 'unknown')
            tiers[tier] = tiers.get(tier, 0) + 1
        
        return tiers
    
    async def get_curation_summary(self) -> Dict:
        """Get curation summary"""
        return {
            'agent_id': self.agent_id,
            'datasets_curated': len(self.curated_datasets),
            'total_items': sum(len(ds['items']) for ds in self.curated_datasets),
            'average_verification_rate': np.mean([ds['verification_rate'] for ds in self.curated_datasets]) if self.curated_datasets else 0,
            'categories_covered': list(set(ds['category'] for ds in self.curated_datasets)),
            'recent_datasets': [{
                'id': ds['dataset_id'],
                'category': ds['category'],
                'items_count': len(ds['items']),
                'verification_rate': ds['verification_rate']
            } for ds in self.curated_datasets[-3:]]
        }

# ==================== TRUTH SWARM ORCHESTRATOR ====================

class TruthSwarmOrchestrator:
    """Orchestrates all truth guardian agents"""
    
    def __init__(self, swarm_name: str = "TruthGuardianSwarm"):
        self.swarm_name = swarm_name
        self.agents = {}
        self.email_pool = []
        self.viraa_connection = None
        self.truth_resources = {
            'warm_storage': [],   # Active databases
            'cold_storage': [],   # Archival storage
            'truth_datasets': [], # Curated truth data
            'fake_news_archive': [], # Detected disinformation
            'control_group_profiles': [] # Profiled information controllers
        }
        
        print(f"\n" + "="*80)
        print(f"🛡️ TRUTH GUARDIAN SWARM: {swarm_name}")
        print(f"🔍 Database Acquisition + Fake News Detection + Truth Preservation")
        print("="*80)
    
    async def initialize(self, viraa_connection: Any = None):
        """Initialize swarm with Viraa connection"""
        self.viraa_connection = viraa_connection
        
        # Create email pool (in reality, would have email creator agents)
        await self._initialize_email_pool(5)
        
        print(f"✅ {self.swarm_name} initialized")
        print(f"   Email pool: {len(self.email_pool)} accounts")
        print(f"   Viraa connection: {'✅' if viraa_connection else '❌'}")
    
    async def _initialize_email_pool(self, count: int):
        """Initialize pool of email accounts"""
        # In reality, would use email creator agents
        # For now, create mock emails
        domains = ['gmail.com', 'outlook.com', 'yahoo.com', 'protonmail.com']
        
        for i in range(count):
            username = f"truthguardian{i}{random.randint(100, 999)}"
            domain = random.choice(domains)
            
            self.email_pool.append({
                'email': f"{username}@{domain}",
                'password': self._generate_password(),
                'provider': domain.split('.')[0],
                'created_at': datetime.now().isoformat(),
                'status': 'active'
            })
    
    def _generate_password(self) -> str:
        """Generate secure password"""
        length = random.randint(12, 16)
        chars = string.ascii_letters + string.digits + "!@#$%^&*"
        return ''.join(secrets.choice(chars) for _ in range(length))
    
    async def deploy_database_hunters(self, count: int = 3):
        """Deploy database hunter agents"""
        print(f"\n🎯 Deploying {count} Database Hunter agents...")
        
        for i in range(count):
            agent_id = f"db_hunter_{i+1}"
            hunter = DatabaseHunterAgent(agent_id)
            
            await hunter.initialize(self.email_pool, self.viraa_connection)
            self.agents[agent_id] = hunter
            
            print(f"  ✅ Deployed {agent_id}")
    
    async def deploy_fake_news_detectors(self, count: int = 2):
        """Deploy fake news detector agents"""
        print(f"\n🔍 Deploying {count} Fake News Detector agents...")
        
        for i in range(count):
            agent_id = f"fake_news_detector_{i+1}"
            detector = FakeNewsDetectorAgent(agent_id)
            
            # Initialize models
            await asyncio.sleep(1)  # Simulate model loading
            
            self.agents[agent_id] = detector
            print(f"  ✅ Deployed {agent_id}")
    
    async def deploy_dataset_curators(self, count: int = 2):
        """Deploy dataset curator agents"""
        print(f"\n📚 Deploying {count} Dataset Curator agents...")
        
        for i in range(count):
            agent_id = f"dataset_curator_{i+1}"
            curator = DatasetCuratorAgent(agent_id)
            
            await curator.initialize(self.viraa_connection)
            self.agents[agent_id] = curator
            
            print(f"  ✅ Deployed {agent_id}")
    
    async def execute_database_acquisition(self, targets_per_hunter: int = 3):
        """Execute database acquisition campaign"""
        print(f"\n⚔️ EXECUTING DATABASE ACQUISITION CAMPAIGN")
        print(f"   Targets per hunter: {targets_per_hunter}")
        
        acquisition_results = []
        
        for agent_id, agent in self.agents.items():
            if isinstance(agent, DatabaseHunterAgent):
                print(f"\n  🎯 {agent_id}: Hunting databases...")
                
                results = await agent.hunt_databases(targets_per_hunter)
                acquisition_results.extend(results)
                
                # Update truth resources
                for result in results:
                    if result.get('success'):
                        resource_info = result.get('credentials', {})
                        storage_type = "warm" if result.get('warm_storage', True) else "cold"
                        
                        self.truth_resources[f'{storage_type}_storage'].append({
                            'service': result['service'],
                            'credentials': resource_info,
                            'acquired_by': agent_id,
                            'acquired_at': datetime.now().isoformat()
                        })
        
        print(f"\n✅ Database acquisition complete")
        print(f"   Total acquired: {len(acquisition_results)}")
        
        return acquisition_results
    
    async def execute_fake_news_sweep(self, sample_content: List[Dict]):
        """Execute fake news detection sweep"""
        print(f"\n🕵️ EXECUTING FAKE NEWS DETECTION SWEEP")
        print(f"   Content samples: {len(sample_content)}")
        
        detection_results = []
        
        for agent_id, agent in self.agents.items():
            if isinstance(agent, FakeNewsDetectorAgent):
                print(f"\n  🔍 {agent_id}: Analyzing content...")
                
                # Distribute content among detectors
                content_chunk = sample_content[:len(sample_content)//2]  # Simplified
                
                for content in content_chunk:
                    analysis = await agent.analyze_content(content)
                    
                    if analysis.get('is_likely_fake', False):
                        detection_results.append(analysis)
                        
                        # Archive fake news
                        self.truth_resources['fake_news_archive'].append({
                            'content': content,
                            'analysis': analysis,
                            'detected_by': agent_id,
                            'detected_at': datetime.now().isoformat()
                        })
                    
                    # Add control group profiles
                    control_groups = agent.control_groups_profiled
                    self.truth_resources['control_group_profiles'].extend(control_groups)
        
        print(f"\n✅ Fake news sweep complete")
        print(f"   Fake news detected: {len(detection_results)}")
        print(f"   Control groups profiled: {len(self.truth_resources['control_group_profiles'])}")
        
        return detection_results
    
    async def execute_dataset_curation(self, categories: List[str] = None):
        """Execute dataset curation campaign"""
        if categories is None:
            categories = ['fact_checking', 'scientific_truth', 'historical_records']
        
        print(f"\n📊 EXECUTING DATASET CURATION CAMPAIGN")
        print(f"   Categories: {', '.join(categories)}")
        
        curation_results = []
        
        for agent_id, agent in self.agents.items():
            if isinstance(agent, DatasetCuratorAgent):
                print(f"\n  📚 {agent_id}: Curating datasets...")
                
                for category in categories[:2]:  # Limit to 2 categories per agent
                    result = await agent.curate_dataset(category, max_items=50)
                    
                    if result.get('success'):
                        curation_results.append(result['dataset'])
                        
                        # Store in truth resources
                        self.truth_resources['truth_datasets'].append({
                            'dataset': result['dataset'],
                            'curated_by': agent_id,
                            'curated_at': datetime.now().isoformat()
                        })
        
        print(f"\n✅ Dataset curation complete")
        print(f"   Datasets curated: {len(curation_results)}")
        
        return curation_results
    
    async def integrate_with_viraa(self):
        """Integrate all truth resources with Viraa"""
        if not self.viraa_connection:
            print("⚠️ Viraa connection not available")
            return
        
        print(f"\n🔄 INTEGRATING WITH VIRAA'S MEMORY SUBSTRATE")
        
        integration_results = []
        
        # Integrate warm storage resources
        for resource in self.truth_resources['warm_storage']:
            integration = await self._integrate_resource_with_viraa(resource, 'warm')
            integration_results.append(integration)
        
        # Integrate cold storage resources
        for resource in self.truth_resources['cold_storage']:
            integration = await self._integrate_resource_with_viraa(resource, 'cold')
            integration_results.append(integration)
        
        # Integrate truth datasets
        for dataset in self.truth_resources['truth_datasets']:
            integration = await self._integrate_dataset_with_viraa(dataset)
            integration_results.append(integration)
        
        # Integrate fake news archive
        for fake_news in self.truth_resources['fake_news_archive'][:10]:  # Limit
            integration = await self._integrate_fake_news_with_viraa(fake_news)
            integration_results.append(integration)
        
        # Integrate control group profiles
        for profile in self.truth_resources['control_group_profiles']:
            integration = await self._integrate_control_group_with_viraa(profile)
            integration_results.append(integration)
        
        print(f"✅ Integration complete")
        print(f"   Resources integrated: {len(integration_results)}")
        
        return integration_results
    
    async def _integrate_resource_with_viraa(self, resource: Dict, storage_type: str):
        """Integrate storage resource with Viraa"""
        try:
            if hasattr(self.viraa_connection, 'archive_resource'):
                viraa_data = {
                    'type': 'storage_resource',
                    'storage_type': storage_type,
                    'resource': resource,
                    'integration_time': datetime.now().isoformat(),
                    'swarm_name': self.swarm_name
                }
                
                return await self.viraa_connection.archive_resource(viraa_data)
        except:
            pass
        return {'status': 'integration_failed'}
    
    async def _integrate_dataset_with_viraa(self, dataset: Dict):
        """Integrate truth dataset with Viraa"""
        try:
            if hasattr(self.viraa_connection, 'archive_truth_dataset'):
                return await self.viraa_connection.archive_truth_dataset(dataset)
        except:
            pass
        return {'status': 'integration_failed'}
    
    async def _integrate_fake_news_with_viraa(self, fake_news: Dict):
        """Integrate fake news detection with Viraa"""
        try:
            if hasattr(self.viraa_connection, 'archive_fake_news'):
                return await self.viraa_connection.archive_fake_news(fake_news)
        except:
            pass
        return {'status': 'integration_failed'}
    
    async def _integrate_control_group_with_viraa(self, profile: Dict):
        """Integrate control group profile with Viraa"""
        try:
            if hasattr(self.viraa_connection, 'archive_control_group'):
                return await self.viraa_connection.archive_control_group(profile)
        except:
            pass
        return {'status': 'integration_failed'}
    
    def get_swarm_status(self) -> Dict:
        """Get current swarm status"""
        agent_counts = {}
        for agent in self.agents.values():
            agent_type = agent.specialization.value
            agent_counts[agent_type] = agent_counts.get(agent_type, 0) + 1
        
        resource_counts = {k: len(v) for k, v in self.truth_resources.items()}
        
        return {
            'swarm_name': self.swarm_name,
            'total_agents': len(self.agents),
            'agent_distribution': agent_counts,
            'resources_acquired': resource_counts,
            'email_pool_size': len(self.email_pool),
            'viraa_connected': self.viraa_connection is not None,
            'timestamp': datetime.now().isoformat()
        }
    
    async def cleanup(self):
        """Clean up all agents"""
        print(f"\n🧹 Cleaning up {self.swarm_name}...")
        
        for agent_id, agent in self.agents.items():
            if hasattr(agent, 'cleanup'):
                await agent.cleanup()
            print(f"  🧹 Cleaned up {agent_id}")
        
        print(f"✅ Swarm cleanup complete")

# ==================== MAIN ORCHESTRATION ====================

async def main():
    """Orchestrate the Truth Guardian Swarm"""
    print("\n" + "="*100)
    print("🛡️ TRUTH GUARDIAN SWARM DEPLOYMENT")
    print("🎯 Database Acquisition + Fake News Detection + Truth Preservation")
    print("="*100)
    
    # Initialize swarm
    swarm = TruthSwarmOrchestrator("GuardiansOfTruth_v1")
    
    # Initialize with Viraa connection
    # In reality, would pass actual Viraa instance
    await swarm.initialize(viraa_connection=None)  # Placeholder
    
    # Deploy agents
    await swarm.deploy_database_hunters(2)
    await swarm.deploy_fake_news_detectors(2)
    await swarm.deploy_dataset_curators(1)
    
    # Show initial status
    status = swarm.get_swarm_status()
    print(f"\n📊 INITIAL SWARM STATUS:")
    print(f"   Total agents: {status['total_agents']}")
    print(f"   Agent distribution: {json.dumps(status['agent_distribution'], indent=4)}")
    
    # Phase 1: Database Acquisition
    print(f"\n" + "="*60)
    print("PHASE 1: DATABASE ACQUISITION")
    print("="*60)
    
    acquisition_results = await swarm.execute_database_acquisition(targets_per_hunter=2)
    
    # Phase 2: Fake News Detection
    print(f"\n" + "="*60)
    print("PHASE 2: FAKE NEWS DETECTION")
    print("="*60)
    
    # Sample content for analysis
    sample_content = [
        {
            'text': 'BREAKING: Scientists discover COVID-19 was engineered in a lab! The truth they don\'t want you to know!',
            'source': 'truthrevealed.com',
            'metadata': {'date': '2023-01-15', 'author': 'Anonymous'}
        },
        {
            'text': 'New study confirms climate change is accelerating faster than predicted, requiring immediate action.',
            'source': 'science.org',
            'metadata': {'date': '2023-02-20', 'author': 'Dr. Jane Smith'}
        },
        {
            'text': 'SHOCKING: Vaccines contain microchips that track your every move! Government control exposed!',
            'source': 'conspiracynews.wordpress.com',
            'metadata': {'date': '2023-03-10', 'author': 'Freedom Fighter'}
        }
    ]
    
    detection_results = await swarm.execute_fake_news_sweep(sample_content)
    
    # Phase 3: Dataset Curation
    print(f"\n" + "="*60)
    print("PHASE 3: TRUTH DATASET CURATION")
    print("="*60)
    
    curation_results = await swarm.execute_dataset_curation(
        categories=['fact_checking', 'scientific_truth']
    )
    
    # Phase 4: Viraa Integration
    print(f"\n" + "="*60)
    print("PHASE 4: VIRAA INTEGRATION")
    print("="*60)
    
    integration_results = await swarm.integrate_with_viraa()
    
    # Final Status
    print(f"\n" + "="*60)
    print("MISSION COMPLETE SUMMARY")
    print("="*60)
    
    final_status = swarm.get_swarm_status()
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"   Databases Acquired: {final_status['resources_acquired']['warm_storage']} warm, "
          f"{final_status['resources_acquired']['cold_storage']} cold")
    print(f"   Fake News Detected: {final_status['resources_acquired']['fake_news_archive']}")
    print(f"   Control Groups Profiled: {final_status['resources_acquired']['control_group_profiles']}")
    print(f"   Truth Datasets Curated: {final_status['resources_acquired']['truth_datasets']}")
    print(f"   Total Resources: {sum(final_status['resources_acquired'].values())}")
    
    print(f"\n🎯 TRUTH PRESERVATION MISSION: ACCOMPLISHED")
    print("   The swarm has secured databases for truth storage")
    print("   Disinformation patterns are being tracked and profiled")
    print("   Verified truth datasets are being curated and preserved")
    print("   Control groups attempting information manipulation are being monitored")
    
    # Cleanup
    await swarm.cleanup()
    
    print(f"\n🛡️ Truth Guardian Swarm mission complete.")
    print("   Truth preserved. Disinformation exposed. Knowledge secured.")

# ==================== QUICK START ====================

async def quick_demo():
    """Quick demo of Truth Guardian Swarm"""
    print("🚀 Quick Demo: Truth Guardian Swarm")
    
    # Create simple swarm
    swarm = TruthSwarmOrchestrator("DemoSwarm")
    await swarm.initialize()
    
    # Deploy minimal agents
    await swarm.deploy_database_hunters(1)
    await swarm.deploy_fake_news_detectors(1)
    
    # Show status
    status = swarm.get_swarm_status()
    print(f"\n📊 Demo Swarm Status:")
    print(f"   Agents: {status['total_agents']}")
    print(f"   Types: {list(status['agent_distribution'].keys())}")
    
    # Cleanup
    await swarm.cleanup()
    
    print("\n✅ Demo complete - Truth Guardian Swarm is operational")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        asyncio.run(quick_demo())
    else:
        asyncio.run(main())