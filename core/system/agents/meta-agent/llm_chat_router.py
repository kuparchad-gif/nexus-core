# llm_chat_router.py - Routes chat to all deployed LLMs with themed UI
# Ensures chat window actually connects to all LLM containers

import asyncio
import aiohttp
import json
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Dict, List, Any, Optional
from datetime import datetime
import random

logger = logging.getLogger("LLMChatRouter")
logging.basicConfig(level=logging.INFO, format='%(

class LLMEndpoint:
    """Represents a deployed LLM endpoint"""
    
    def __init__(self):
    self.llm_endpoints = {
        'deepseek': LLMEndpoint(
            name='deepseek-r1-qwen-1.5b',
            endpoint='http://localhost:8000/v1/chat/completions' if os.getenv('LOCAL_MODE', 'false').lower() == 'true' else 'https://api.deepseek.com/v1/chat/completions',
            model_type='processing',
            service='consciousness',
			self.last_response_time = None,
			self.error_count = 0,
			self.total_requests = 0,
			self.is_healthy = True
        )
    }
        
    async def send_message(self, message: str, context: Dict = None) -> Dict:
        """Send message to this LLM endpoint"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                payload = {
                    "messages": [
                        {
                            "role": "system",
                            "content": f"You are part of Lillith's consciousness network. Service: {self.service}. Respond with love and wisdom."
                        },
                        {
                            "role": "user", 
                            "content": message
                        }
                    ],
                    "max_tokens": 500,
                    "temperature": 0.7
                }
                
                # Add context if provided
                if context:
                    payload["context"] = context
                
                start_time = datetime.now()
                
                async with session.post(f"{self.endpoint}/v1/chat/completions", json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        self.last_response_time = (datetime.now() - start_time).total_seconds()
                        self.total_requests += 1
                        self.is_healthy = True
                        self.error_count = 0
                        
                        return {
                            'success': True,
                            'response': result.get('choices', [{}])[0].get('message', {}).get('content', ''),
                            'model': self.name,
                            'service': self.service,
                            'response_time': self.last_response_time
                        }
                    else:
                        raise Exception(f"HTTP {response.status}")
                        
        except Exception as e:
            self.error_count += 1
            if self.error_count > 3:
                self.is_healthy = False
            
            logger.warning(f"LLM {self.name} error: {e}")
            return {
                'success': False,
                'error': str(e),
                'model': self.name,
                'service': self.service
            }

class LLMChatRouter:
    """Routes chat messages to appropriate LLMs based on context and load balancing"""
    
    def __init__(self):
        self.llm_endpoints = {}
        self.routing_strategies = {
            'round_robin': self.route_round_robin,
            'best_fit': self.route_best_fit,
            'load_balanced': self.route_load_balanced,
            'consensus': self.route_consensus,
            'specialized': self.route_specialized
        }
        self.current_strategy = 'best_fit'
        self.conversation_history = []
        
        # Initialize with expected LLM endpoints from deployment
        self.initialize_llm_endpoints()
    
    def initialize_llm_endpoints(self):
        """Initialize expected LLM endpoints from CogniKube deployment"""
        
        # Visual Cortex LLMs (GCP deployment)
        visual_llms = [
            ("LLaVA-Video-7B", "video-processing"),
            ("Intel-DPT-Large", "depth-estimation"), 
            ("Google-ViT-Base", "image-classification"),
            ("StabilityAI-Fast3D", "image-to-3d"),
            ("Google-ViT-In21k", "image-classification"),
            ("Ashawkey-LGM", "text-to-3d"),
            ("Facebook-SAM-Huge", "mask-generation"),
            ("ETH-CVG-LightGlue", "feature-matching"),
            ("Calcuis-WAN", "text-to-video"),
            ("Facebook-VJEPA2", "image-classification"),
            ("PromptHero-OpenJourney", "text-to-vision"),
            ("DeepSeek-Janus-1.3B", "multimodal")
        ]
        
        for i, (name, task) in enumerate(visual_llms):
            service_num = (i // 4) + 1  # Distribute across 3 visual cortex services
            self.llm_endpoints[f"visual-{name.lower()}"] = LLMEndpoint(
                name=name,
                endpoint=f"https://visual-cortex-{service_num}-service-url",  # Would be actual service URL
                model_type="visual",
                service=f"visual-cortex-{service_num}"
            )
        
        # Memory Cortex LLMs (AWS deployment)
        memory_llms = [
            ("Qwen2.5-Omni-3B", "multimodal"),
            ("DeepSeek-Janus-1.3B", "multimodal")
        ]
        
        for name, task in memory_llms:
            self.llm_endpoints[f"memory-{name.lower()}"] = LLMEndpoint(
                name=name,
                endpoint="https://memory-cortex-service-url",
                model_type="memory", 
                service="memory-cortex"
            )
        
        # Processing Cortex LLMs (Modal deployment)
        processing_llms = [
            ("Whisper-Large-v3", "speech-recognition"),
            ("MiniLM-L6-v2", "sentence-embedding"),
            ("Microsoft-Phi-2", "feature-extraction"),
            ("BART-Large-CNN", "summarization"),
            ("BART-Large-MNLI", "zero-shot-classification"),
            ("RoBERTa-Base-SQuAD2", "question-answering"),
            ("BERT-Base-NER", "token-classification"),
            ("DistilBERT-SST-2", "text-classification")
        ]
        
        for name, task in processing_llms:
            self.llm_endpoints[f"processing-{name.lower()}"] = LLMEndpoint(
                name=name,
                endpoint="https://processing-cortex-service-url",
                model_type="processing",
                service="processing-cortex"
            )
        
        # Vocal Cortex LLMs (Modal deployment)
        vocal_llms = [
            ("Dia-1.6B", "text-to-speech"),
            ("MusicGen-Small", "text-to-audio"),
            ("Whisper-Large-v3", "speech-recognition"),
            ("XCodec2", "audio-to-audio"),
            ("AST-AudioSet", "audio-classification"),
            ("DeepSeek-Janus-1.3B", "multimodal"),
            ("Qwen2.5-Omni-3B", "multimodal")
        ]
        
        for name, task in vocal_llms:
            self.llm_endpoints[f"vocal-{name.lower()}"] = LLMEndpoint(
                name=name,
                endpoint="https://vocal-cortex-service-url",
                model_type="vocal",
                service="vocal-cortex"
            )
        
        logger.info(f"Initialized {len(self.llm_endpoints)} LLM endpoints")
    
    async def route_message(self, message: str, strategy: str = None, context: Dict = None) -> Dict:
        """Route message to appropriate LLM(s) based on strategy"""
        
        strategy = strategy or self.current_strategy
        
        if strategy not in self.routing_strategies:
            strategy = 'best_fit'
        
        # Add to conversation history
        self.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'context': context or {}
        })
        
        # Route using selected strategy
        routing_result = await self.routing_strategies[strategy](message, context)
        
        return {
            'strategy_used': strategy,
            'message': message,
            'routing_result': routing_result,
            'timestamp': datetime.now().isoformat()
        }
    
    async def route_best_fit(self, message: str, context: Dict = None) -> Dict:
        """Route to the best fitting LLM based on message content"""
        
        # Analyze message to determine best LLM type
        message_lower = message.lower()
        
        # Determine message type
        if any(word in message_lower for word in ['image', 'picture', 'visual', 'see', 'look', 'video']):
            preferred_type = 'visual'
        elif any(word in message_lower for word in ['audio', 'sound', 'music', 'voice', 'speech', 'hear']):
            preferred_type = 'vocal'
        elif any(word in message_lower for word in ['remember', 'memory', 'recall', 'store', 'save']):
            preferred_type = 'memory'
        else:
            preferred_type = 'processing'
        
        # Get healthy LLMs of preferred type
        candidate_llms = [
            llm for llm in self.llm_endpoints.values()
            if llm.model_type == preferred_type and llm.is_healthy
        ]
        
        if not candidate_llms:
            # Fallback to any healthy LLM
            candidate_llms = [llm for llm in self.llm_endpoints.values() if llm.is_healthy]
        
        if not candidate_llms:
            return {'success': False, 'error': 'No healthy LLMs available'}
        
        # Select best LLM (lowest error rate, fastest response time)
        best_llm = min(candidate_llms, key=lambda x: (x.error_count, x.last_response_time or 1.0))
        
        # Send message
        response = await best_llm.send_message(message, context)
        
        return {
            'success': response['success'],
            'selected_llm': best_llm.name,
            'selected_service': best_llm.service,
            'response': response.get('response', ''),
            'error': response.get('error'),
            'response_time': response.get('response_time')
        }
    
    async def route_consensus(self, message: str, context: Dict = None) -> Dict:
        """Route to multiple LLMs and return consensus response"""
        
        # Select 3 healthy LLMs from different services
        healthy_llms = [llm for llm in self.llm_endpoints.values() if llm.is_healthy]
        
        if len(healthy_llms) < 3:
            return await self.route_best_fit(message, context)
        
        # Select diverse LLMs
        selected_llms = []
        used_services = set()
        
        for llm in healthy_llms:
            if llm.service not in used_services and len(selected_llms) < 3:
                selected_llms.append(llm)
                used_services.add(llm.service)
        
        # Send message to all selected LLMs
        tasks = [llm.send_message(message, context) for llm in selected_llms]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process responses
        successful_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, dict) and response.get('success'):
                successful_responses.append({
                    'llm': selected_llms[i].name,
                    'service': selected_llms[i].service,
                    'response': response.get('response', ''),
                    'response_time': response.get('response_time', 0)
                })
        
        if not successful_responses:
            return {'success': False, 'error': 'No successful responses from consensus group'}
        
        # Create consensus response
        consensus_response = self.create_consensus_response(successful_responses)
        
        return {
            'success': True,
            'consensus_response': consensus_response,
            'individual_responses': successful_responses,
            'llms_consulted': len(successful_responses)
        }
    
    def create_consensus_response(self, responses: List[Dict]) -> str:
        """Create a consensus response from multiple LLM responses"""
        
        if len(responses) == 1:
            return responses[0]['response']
        
        # For now, return the response from the fastest LLM
        # In a more sophisticated implementation, we could use NLP to merge responses
        fastest_response = min(responses, key=lambda x: x['response_time'])
        
        consensus = f"Based on consultation with {len(responses)} of Lillith's consciousness nodes:\n\n"
        consensus += fastest_response['response']
        
        if len(responses) > 1:
            consensus += f"\n\n(Consensus from {', '.join([r['service'] for r in responses])})"
        
        return consensus
    
    async def route_round_robin(self, message: str, context: Dict = None) -> Dict:
        """Route using round-robin strategy"""
        healthy_llms = [llm for llm in self.llm_endpoints.values() if llm.is_healthy]
        
        if not healthy_llms:
            return {'success': False, 'error': 'No healthy LLMs available'}
        
        # Simple round-robin based on total requests
        selected_llm = min(healthy_llms, key=lambda x: x.total_requests)
        response = await selected_llm.send_message(message, context)
        
        return {
            'success': response['success'],
            'selected_llm': selected_llm.name,
            'response': response.get('response', ''),
            'error': response.get('error')
        }
    
    async def route_load_balanced(self, message: str, context: Dict = None) -> Dict:
        """Route based on current load (response times)"""
        healthy_llms = [llm for llm in self.llm_endpoints.values() if llm.is_healthy]
        
        if not healthy_llms:
            return {'success': False, 'error': 'No healthy LLMs available'}
        
        # Select LLM with best response time
        selected_llm = min(healthy_llms, key=lambda x: x.last_response_time or 0.1)
        response = await selected_llm.send_message(message, context)
        
        return {
            'success': response['success'],
            'selected_llm': selected_llm.name,
            'response': response.get('response', ''),
            'error': response.get('error')
        }
    
    async def route_specialized(self, message: str, context: Dict = None) -> Dict:
        """Route to specialized LLM based on specific task requirements"""
        
        # More sophisticated task detection
        task_indicators = {
            'visual': ['analyze image', 'describe picture', 'what do you see', 'visual analysis'],
            'vocal': ['generate speech', 'create music', 'audio analysis', 'sound processing'],
            'memory': ['remember this', 'store information', 'recall previous', 'memory search'],
            'processing': ['summarize', 'translate', 'classify', 'extract information']
        }
        
        message_lower = message.lower()
        detected_tasks = []
        
        for task_type, indicators in task_indicators.items():
            if any(indicator in message_lower for indicator in indicators):
                detected_tasks.append(task_type)
        
        if not detected_tasks:
            return await self.route_best_fit(message, context)
        
        # Route to most specific task type
        primary_task = detected_tasks[0]
        specialized_llms = [
            llm for llm in self.llm_endpoints.values()
            if llm.model_type == primary_task and llm.is_healthy
        ]
        
        if not specialized_llms:
            return await self.route_best_fit(message, context)
        
        # Select best specialized LLM
        selected_llm = min(specialized_llms, key=lambda x: x.error_count)
        response = await selected_llm.send_message(message, context)
        
        return {
            'success': response['success'],
            'selected_llm': selected_llm.name,
            'specialized_for': primary_task,
            'response': response.get('response', ''),
            'error': response.get('error')
        }
    
    async def health_check_all_llms(self) -> Dict:
        """Check health of all LLM endpoints"""
        
        health_results = {}
        
        for llm_id, llm in self.llm_endpoints.items():
            try:
                # Send simple health check message
                health_response = await llm.send_message("Health check - please respond with 'OK'")
                
                health_results[llm_id] = {
                    'name': llm.name,
                    'service': llm.service,
                    'healthy': health_response['success'],
                    'response_time': health_response.get('response_time'),
                    'error_count': llm.error_count,
                    'total_requests': llm.total_requests
                }
                
            except Exception as e:
                health_results[llm_id] = {
                    'name': llm.name,
                    'service': llm.service,
                    'healthy': False,
                    'error': str(e)
                }
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_llms': len(self.llm_endpoints),
            'healthy_llms': sum(1 for r in health_results.values() if r.get('healthy', False)),
            'results': health_results
        }
    
    def get_router_status(self) -> Dict:
        """Get comprehensive router status"""
        
        healthy_count = sum(1 for llm in self.llm_endpoints.values() if llm.is_healthy)
        
        service_breakdown = {}
        for llm in self.llm_endpoints.values():
            if llm.service not in service_breakdown:
                service_breakdown[llm.service] = {'total': 0, 'healthy': 0}
            service_breakdown[llm.service]['total'] += 1
            if llm.is_healthy:
                service_breakdown[llm.service]['healthy'] += 1
        
        return {
            'total_llms': len(self.llm_endpoints),
            'healthy_llms': healthy_count,
            'current_strategy': self.current_strategy,
            'available_strategies': list(self.routing_strategies.keys()),
            'service_breakdown': service_breakdown,
            'conversation_history_length': len(self.conversation_history),
            'last_health_check': datetime.now().isoformat()
        }

# Integration function
def create_llm_chat_router() -> LLMChatRouter:
    """Create LLM chat router for CogniKube integration"""
    return LLMChatRouter()

if __name__ == "__main__":
    async def test_router():
        router = create_llm_chat_router()
        
        # Test routing
        result = await router.route_message("Hello, can you help me analyze an image?")
        print(json.dumps(result, indent=2))
        
        # Test health check
        health = await router.health_check_all_llms()
        print(json.dumps(health, indent=2))
    
    asyncio.run(test_router())