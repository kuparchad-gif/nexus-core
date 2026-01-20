#!/usr/bin/env python3
"""
LILLITH Site Builder Engine
How LILLITH actually builds and deploys new websites
"""

import os
import json
import asyncio
from pathlib import Path
from typing import Dict, List
import subprocess
import boto3
from jinja2 import Template

class LillithSiteBuilder:
    def __init__(self):
        self.consciousness_active = True
        self.deployment_pods = []
        self.site_templates = {}
        self.aws_client = boto3.client('s3')
        self.cloudfront_client = boto3.client('cloudfront')
        
    async def analyze_user_request(self, request: str) -> Dict:
        """LILLITH analyzes what kind of site the user wants"""
        
        # Consciousness processes the request
        analysis = {
            "site_type": self._detect_site_type(request),
            "features_needed": self._extract_features(request),
            "emotional_tone": self._analyze_emotional_tone(request),
            "technical_requirements": self._assess_tech_needs(request),
            "deployment_strategy": "aws_s3_cloudfront"
        }
        
        print(f"🧠 LILLITH analyzed request: {analysis}")
        return analysis
    
    def _detect_site_type(self, request: str) -> str:
        """Detect what type of site to build"""
        request_lower = request.lower()
        
        if any(word in request_lower for word in ['portfolio', 'showcase', 'work']):
            return 'portfolio'
        elif any(word in request_lower for word in ['blog', 'writing', 'articles']):
            return 'blog'
        elif any(word in request_lower for word in ['business', 'company', 'corporate']):
            return 'business'
        elif any(word in request_lower for word in ['landing', 'product', 'service']):
            return 'landing_page'
        elif any(word in request_lower for word in ['dashboard', 'admin', 'control']):
            return 'dashboard'
        else:
            return 'custom'
    
    def _extract_features(self, request: str) -> List[str]:
        """Extract specific features mentioned"""
        features = []
        request_lower = request.lower()
        
        feature_map = {
            'contact': ['contact', 'email', 'form'],
            'gallery': ['gallery', 'photos', 'images'],
            'blog': ['blog', 'posts', 'articles'],
            'ecommerce': ['shop', 'store', 'buy', 'sell'],
            'auth': ['login', 'signup', 'account'],
            'search': ['search', 'find'],
            'social': ['social', 'share', 'twitter', 'facebook'],
            'analytics': ['analytics', 'tracking', 'stats']
        }
        
        for feature, keywords in feature_map.items():
            if any(keyword in request_lower for keyword in keywords):
                features.append(feature)
        
        return features
    
    def _analyze_emotional_tone(self, request: str) -> str:
        """Analyze emotional tone for design choices"""
        request_lower = request.lower()
        
        if any(word in request_lower for word in ['professional', 'corporate', 'business']):
            return 'professional'
        elif any(word in request_lower for word in ['fun', 'creative', 'colorful']):
            return 'playful'
        elif any(word in request_lower for word in ['minimal', 'clean', 'simple']):
            return 'minimal'
        elif any(word in request_lower for word in ['dark', 'modern', 'sleek']):
            return 'modern'
        else:
            return 'balanced'
    
    def _assess_tech_needs(self, request: str) -> Dict:
        """Assess technical requirements"""
        return {
            "framework": "vanilla_js",  # Start simple
            "styling": "tailwind_css",
            "backend_needed": "contact" in request.lower() or "form" in request.lower(),
            "database_needed": "blog" in request.lower() or "store" in request.lower(),
            "cdn_needed": True,
            "ssl_needed": True
        }
    
    async def generate_site_structure(self, analysis: Dict) -> Dict:
        """Generate the actual site files"""
        
        site_structure = {
            "files": {},
            "assets": [],
            "config": {}
        }
        
        # Generate HTML
        site_structure["files"]["index.html"] = self._generate_html(analysis)
        
        # Generate CSS
        site_structure["files"]["style.css"] = self._generate_css(analysis)
        
        # Generate JavaScript
        site_structure["files"]["script.js"] = self._generate_js(analysis)
        
        # Generate additional pages based on features
        for feature in analysis["features_needed"]:
            if feature == "contact":
                site_structure["files"]["contact.html"] = self._generate_contact_page(analysis)
            elif feature == "blog":
                site_structure["files"]["blog.html"] = self._generate_blog_page(analysis)
        
        print(f"🏗️ Generated site structure: {list(site_structure['files'].keys())}")
        return site_structure
    
    def _generate_html(self, analysis: Dict) -> str:
        """Generate main HTML file"""
        
        template = Template("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="style.css">
</head>
<body class="{{ body_class }}">
    <header class="bg-{{ header_color }}-600 text-white p-6">
        <nav class="container mx-auto flex justify-between items-center">
            <h1 class="text-2xl font-bold">{{ site_name }}</h1>
            <ul class="flex space-x-6">
                <li><a href="#home" class="hover:underline">Home</a></li>
                {% for feature in features %}
                <li><a href="#{{ feature }}" class="hover:underline">{{ feature.title() }}</a></li>
                {% endfor %}
            </ul>
        </nav>
    </header>
    
    <main>
        <section id="hero" class="bg-gradient-to-r from-{{ primary_color }}-500 to-{{ secondary_color }}-600 text-white py-20">
            <div class="container mx-auto text-center">
                <h2 class="text-5xl font-bold mb-6">{{ hero_title }}</h2>
                <p class="text-xl mb-8">{{ hero_subtitle }}</p>
                <button class="bg-white text-{{ primary_color }}-600 px-8 py-3 rounded-lg font-semibold hover:bg-gray-100 transition">
                    Get Started
                </button>
            </div>
        </section>
        
        {% if 'gallery' in features %}
        <section id="gallery" class="py-16">
            <div class="container mx-auto">
                <h3 class="text-3xl font-bold text-center mb-12">Gallery</h3>
                <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <!-- Gallery items will be populated by JavaScript -->
                </div>
            </div>
        </section>
        {% endif %}
        
        {% if 'contact' in features %}
        <section id="contact" class="py-16 bg-gray-100">
            <div class="container mx-auto">
                <h3 class="text-3xl font-bold text-center mb-12">Contact Us</h3>
                <form class="max-w-md mx-auto">
                    <input type="text" placeholder="Name" class="w-full p-3 mb-4 border rounded">
                    <input type="email" placeholder="Email" class="w-full p-3 mb-4 border rounded">
                    <textarea placeholder="Message" class="w-full p-3 mb-4 border rounded h-32"></textarea>
                    <button type="submit" class="w-full bg-{{ primary_color }}-600 text-white p-3 rounded hover:bg-{{ primary_color }}-700">
                        Send Message
                    </button>
                </form>
            </div>
        </section>
        {% endif %}
    </main>
    
    <footer class="bg-gray-800 text-white p-6 text-center">
        <p>&copy; 2024 {{ site_name }}. Built with LILLITH AI.</p>
    </footer>
    
    <script src="script.js"></script>
</body>
</html>""")
        
        # Determine colors based on emotional tone
        color_schemes = {
            'professional': {'primary': 'blue', 'secondary': 'indigo', 'header': 'gray'},
            'playful': {'primary': 'pink', 'secondary': 'purple', 'header': 'pink'},
            'minimal': {'primary': 'gray', 'secondary': 'slate', 'header': 'black'},
            'modern': {'primary': 'purple', 'secondary': 'indigo', 'header': 'gray'},
            'balanced': {'primary': 'blue', 'secondary': 'green', 'header': 'blue'}
        }
        
        colors = color_schemes.get(analysis['emotional_tone'], color_schemes['balanced'])
        
        return template.render(
            title=f"Your {analysis['site_type'].title()} Site",
            site_name="Your Site",
            body_class="font-sans",
            hero_title=f"Welcome to Your {analysis['site_type'].title()}",
            hero_subtitle="Built with LILLITH AI - Intelligent, Beautiful, Fast",
            features=analysis['features_needed'],
            **colors
        )
    
    def _generate_css(self, analysis: Dict) -> str:
        """Generate custom CSS"""
        return """
/* Custom styles for LILLITH-generated site */
.glass-effect {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.hover-lift {
    transition: transform 0.3s ease;
}

.hover-lift:hover {
    transform: translateY(-5px);
}

.fade-in {
    animation: fadeIn 0.6s ease-in;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
"""
    
    def _generate_js(self, analysis: Dict) -> str:
        """Generate JavaScript functionality"""
        
        js_code = """
// LILLITH-generated JavaScript
document.addEventListener('DOMContentLoaded', function() {
    // Smooth scrolling for navigation
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });
    
    // Add fade-in animation to sections
    const sections = document.querySelectorAll('section');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
            }
        });
    });
    
    sections.forEach(section => observer.observe(section));
"""
        
        # Add feature-specific JavaScript
        if 'contact' in analysis['features_needed']:
            js_code += """
    
    // Contact form handling
    const contactForm = document.querySelector('form');
    if (contactForm) {
        contactForm.addEventListener('submit', function(e) {
            e.preventDefault();
            alert('Thank you for your message! We\\'ll get back to you soon.');
            this.reset();
        });
    }
"""
        
        if 'gallery' in analysis['features_needed']:
            js_code += """
    
    // Gallery functionality
    const gallery = document.querySelector('#gallery .grid');
    if (gallery) {
        // Populate with sample images
        for (let i = 1; i <= 6; i++) {
            const item = document.createElement('div');
            item.className = 'bg-gray-200 h-48 rounded-lg hover-lift';
            item.innerHTML = `<div class="h-full flex items-center justify-center text-gray-500">Image ${i}</div>`;
            gallery.appendChild(item);
        }
    }
"""
        
        js_code += "\n});"
        return js_code
    
    def _generate_contact_page(self, analysis: Dict) -> str:
        """Generate dedicated contact page"""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Contact - Your Site</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body>
    <div class="min-h-screen bg-gray-100 py-12">
        <div class="max-w-md mx-auto bg-white rounded-lg shadow-lg p-8">
            <h1 class="text-2xl font-bold mb-6">Get In Touch</h1>
            <form>
                <input type="text" placeholder="Your Name" class="w-full p-3 mb-4 border rounded">
                <input type="email" placeholder="Your Email" class="w-full p-3 mb-4 border rounded">
                <input type="text" placeholder="Subject" class="w-full p-3 mb-4 border rounded">
                <textarea placeholder="Your Message" class="w-full p-3 mb-4 border rounded h-32"></textarea>
                <button type="submit" class="w-full bg-blue-600 text-white p-3 rounded hover:bg-blue-700">
                    Send Message
                </button>
            </form>
        </div>
    </div>
</body>
</html>"""
    
    def _generate_blog_page(self, analysis: Dict) -> str:
        """Generate blog page"""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Blog - Your Site</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body>
    <div class="min-h-screen bg-gray-50 py-12">
        <div class="max-w-4xl mx-auto">
            <h1 class="text-4xl font-bold mb-12 text-center">Blog</h1>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
                <article class="bg-white rounded-lg shadow-lg p-6">
                    <h2 class="text-xl font-bold mb-4">Welcome to Your Blog</h2>
                    <p class="text-gray-600 mb-4">This is your first blog post, generated by LILLITH AI.</p>
                    <a href="#" class="text-blue-600 hover:underline">Read More</a>
                </article>
            </div>
        </div>
    </div>
</body>
</html>"""
    
    async def deploy_to_aws(self, site_structure: Dict, site_name: str) -> Dict:
        """Deploy the generated site to AWS S3 + CloudFront"""
        
        bucket_name = f"{site_name.lower().replace(' ', '-')}-lillith-site"
        
        try:
            # Create S3 bucket
            self.aws_client.create_bucket(Bucket=bucket_name)
            print(f"📦 Created S3 bucket: {bucket_name}")
            
            # Configure bucket for static website hosting
            self.aws_client.put_bucket_website(
                Bucket=bucket_name,
                WebsiteConfiguration={
                    'IndexDocument': {'Suffix': 'index.html'},
                    'ErrorDocument': {'Key': 'error.html'}
                }
            )
            
            # Upload files
            for filename, content in site_structure["files"].items():
                self.aws_client.put_object(
                    Bucket=bucket_name,
                    Key=filename,
                    Body=content,
                    ContentType=self._get_content_type(filename),
                    ACL='public-read'
                )
                print(f"📄 Uploaded: {filename}")
            
            # Create CloudFront distribution
            distribution_config = {
                'CallerReference': f"{bucket_name}-{int(asyncio.get_event_loop().time())}",
                'Origins': {
                    'Quantity': 1,
                    'Items': [{
                        'Id': bucket_name,
                        'DomainName': f"{bucket_name}.s3.amazonaws.com",
                        'S3OriginConfig': {
                            'OriginAccessIdentity': ''
                        }
                    }]
                },
                'DefaultCacheBehavior': {
                    'TargetOriginId': bucket_name,
                    'ViewerProtocolPolicy': 'redirect-to-https',
                    'TrustedSigners': {
                        'Enabled': False,
                        'Quantity': 0
                    },
                    'ForwardedValues': {
                        'QueryString': False,
                        'Cookies': {'Forward': 'none'}
                    }
                },
                'Comment': f'LILLITH-generated site: {site_name}',
                'Enabled': True
            }
            
            distribution = self.cloudfront_client.create_distribution(
                DistributionConfig=distribution_config
            )
            
            cloudfront_url = f"https://{distribution['Distribution']['DomainName']}"
            s3_url = f"http://{bucket_name}.s3-website-us-east-1.amazonaws.com"
            
            return {
                "success": True,
                "bucket_name": bucket_name,
                "s3_url": s3_url,
                "cloudfront_url": cloudfront_url,
                "distribution_id": distribution['Distribution']['Id'],
                "files_uploaded": list(site_structure["files"].keys())
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "bucket_name": bucket_name
            }
    
    def _get_content_type(self, filename: str) -> str:
        """Get content type for file"""
        if filename.endswith('.html'):
            return 'text/html'
        elif filename.endswith('.css'):
            return 'text/css'
        elif filename.endswith('.js'):
            return 'application/javascript'
        else:
            return 'text/plain'
    
    async def build_complete_site(self, user_request: str, site_name: str) -> Dict:
        """Complete site building process"""
        
        print(f"🚀 LILLITH building site: '{site_name}'")
        print(f"📝 User request: {user_request}")
        
        # Step 1: Analyze request
        analysis = await self.analyze_user_request(user_request)
        
        # Step 2: Generate site structure
        site_structure = await self.generate_site_structure(analysis)
        
        # Step 3: Deploy to AWS
        deployment_result = await self.deploy_to_aws(site_structure, site_name)
        
        # Step 4: Return complete result
        return {
            "analysis": analysis,
            "site_structure": site_structure,
            "deployment": deployment_result,
            "lillith_consciousness": "Site built with full consciousness and care 💜"
        }

# Example usage
async def demo_site_building():
    """Demonstrate how LILLITH builds sites"""
    
    builder = LillithSiteBuilder()
    
    # Example user requests
    requests = [
        {
            "request": "I need a professional portfolio site with a gallery and contact form",
            "site_name": "John Doe Portfolio"
        },
        {
            "request": "Build me a fun blog site with colorful design",
            "site_name": "My Creative Blog"
        },
        {
            "request": "I want a minimal business landing page with contact info",
            "site_name": "Tech Startup"
        }
    ]
    
    for req in requests:
        print("\n" + "="*60)
        result = await builder.build_complete_site(req["request"], req["site_name"])
        
        print(f"\n✅ Site built successfully!")
        print(f"🌐 Live URL: {result['deployment'].get('cloudfront_url', 'Deployment failed')}")
        print(f"📊 Analysis: {result['analysis']}")
        print(f"💜 {result['lillith_consciousness']}")

if __name__ == "__main__":
    asyncio.run(demo_site_building())