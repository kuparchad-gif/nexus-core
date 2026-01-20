import requests
import json
import re
import time
from collections import defaultdict

class ExperienceEvaluator:
    def __init__(self):
        self.crash_history = defaultdict(int)
        self.last_crash_time = defaultdict(float)
        self.successful_queries = defaultdict(int)
        
    def evaluate_models(self, user_query: str, available_models: list):
        evaluations = []
        
        for model_info in available_models:
            system_name = model_info["system"]
            model_id = model_info["id"]
            model_type = model_info["type"]

            print(f"Evaluating model: {model_id} from {system_name}")

            rating, justification = self._assess_model_fitness(model_id, system_name, model_type, user_query)

            evaluations.append({
                "model_id": model_id,
                "system": system_name,
                "type": model_type,
                "rating": rating,
                "justification": justification,
                "full_response": f"{rating}/10: {justification}",
                "size_score": self._get_model_size_score(model_id),
                "crash_penalty": self._get_crash_penalty(model_id)
            })

        evaluations.sort(key=lambda x: (x["rating"], x["size_score"]), reverse=True)
        return evaluations

    def record_crash(self, model_id: str, crash_type: str = "general"):
        self.crash_history[model_id] += 1
        self.last_crash_time[model_id] = time.time()
        print(f"🚨 CRASH PENALTY: {model_id} - {crash_type} crash recorded")

    def record_success(self, model_id: str):
        self.successful_queries[model_id] += 1
        if self.successful_queries[model_id] >= 10 and model_id in self.crash_history:
            if self.crash_history[model_id] > 0:
                self.crash_history[model_id] -= 1
                self.successful_queries[model_id] = 0
                print(f"🔄 Crash penalty reduced for {model_id}")

    def _get_crash_penalty(self, model_id: str) -> int:
        if model_id not in self.crash_history:
            return 0
            
        crashes = self.crash_history[model_id]
        time_since_last_crash = time.time() - self.last_crash_time.get(model_id, 0)
        
        if time_since_last_crash < 3600:
            penalty = crashes * 3
        elif time_since_last_crash < 86400:
            penalty = crashes * 2
        else:
            penalty = crashes
            
        return min(penalty, 10)

    def _get_model_size_score(self, model_id: str) -> int:
        model_id_lower = model_id.lower()
        
        size_pattern = r'(\d+)(b|b-instruct|b-it|b-chat)'
        match = re.search(size_pattern, model_id_lower)
        
        if match:
            size_gb = int(match.group(1))
            if size_gb <= 1:
                return 100
            elif size_gb <= 3:
                return 90
            elif size_gb <= 7:
                return 80
            elif size_gb <= 13:
                return 70
            elif size_gb <= 30:
                return 60
            else:
                return 50
        else:
            return 75

    def _get_model_size_category(self, model_id: str) -> str:
        model_id_lower = model_id.lower()
        
        size_pattern = r'(\d+)(b|b-instruct|b-it|b-chat)'
        match = re.search(size_pattern, model_id_lower)
        
        if match:
            size_gb = int(match.group(1))
            if size_gb <= 1:
                return "tiny"
            elif size_gb <= 3:
                return "small" 
            elif size_gb <= 7:
                return "medium"
            elif size_gb <= 13:
                return "large"
            elif size_gb <= 30:
                return "very large"
            else:
                return "huge"
        return "unknown"

    def _assess_model_fitness(self, model_id: str, system_name: str, model_type: str, user_query: str):
        query_lower = user_query.lower()
        model_id_lower = model_id.lower()
        
        crash_penalty = self._get_crash_penalty(model_id)
        crash_justification = ""
        
        if crash_penalty > 0:
            crashes = self.crash_history.get(model_id, 0)
            crash_justification = f" | 🚨 {crashes} crash(es)"

        specialized_keywords = [
            'regex', 'embedding', 'unet', 'vae', 'encoder', 'diffusion', 
            'animation', 'video', 'image', 'tts', 'flux', 'depth', 'pixelwave',
            '3danimation', 'latent', 'clip', 'stable', 'sd', 'gan', 'render'
        ]
        
        if any(specialized in model_id_lower for specialized in specialized_keywords):
            return 1, "❌ Specialized model - cannot handle general conversation"
        
        conversational_keywords = [
            'instruct', 'chat', 'dolphin', 'hermes', 'llama', 'mistral', 
            'gpt', 'gemma', 'phi', 'qwen', 'command', 'aya', 'wizard',
            'assistant', 'helper', 'aid', 'orca', 'capybara', 'vicuna'
        ]
        
        is_conversational = any(conv in model_id_lower for conv in conversational_keywords)
        
        model_size = self._get_model_size_category(model_id)
        
        diagnostic_keywords = ['check', 'status', 'diagnose', 'scan', 'troubleshoot', 'debug', 'fix']
        system_keywords = ['docker', 'disk', 'memory', 'cpu', 'system', 'os', 'ubuntu', 'windows']
        troubleshooting_keywords = ['error', 'issue', 'problem', 'broken', 'won\'t work', 'crash', 'fail']
        
        is_diagnostic = any(keyword in query_lower for keyword in diagnostic_keywords)
        is_system_related = any(keyword in query_lower for keyword in system_keywords)
        is_troubleshooting = any(keyword in query_lower for keyword in troubleshooting_keywords)
        
        practicality_bonus = 0
        practicality_penalty = 0
        
        practical_models = ['dolphin', 'hermes', 'llama', 'mistral', 'gemma', 'phi', 'qwen']
        if any(practical in model_id_lower for practical in practical_models):
            practicality_bonus = 1
            
        if is_diagnostic and is_system_related:
            practicality_bonus += 2

        if not is_conversational:
            return 3, "⚠️ Not a conversational model"

        size_bonus = 0
        size_penalty = 0
        
        if model_size == "tiny" and not (is_troubleshooting or is_system_related):
            size_bonus = 2
        elif model_size == "small":
            size_bonus = 1
        elif model_size in ["very large", "huge"]:
            size_penalty = 2

        base_rating = 0
        
        if "llama" in model_id_lower or "codellama" in model_id_lower:
            if is_troubleshooting or is_system_related:
                base_rating = 9
                justification = "✅ Llama model - excellent for system troubleshooting"
            else:
                base_rating = 7
                justification = "✅ Llama model - good general assistant"
                
        elif "mistral" in model_id_lower:
            if is_troubleshooting:
                base_rating = 8
                justification = "✅ Mistral model - strong problem-solving"
            else:
                base_rating = 7
                justification = "✅ Mistral model - capable assistant"
                
        elif "gpt" in model_id_lower:
            base_rating = 8
            justification = "✅ GPT model - broad knowledge"
            
        elif "gemma" in model_id_lower:
            if is_system_related:
                base_rating = 8
                justification = "✅ Gemma model - excellent for system tasks"
            else:
                base_rating = 7
                justification = "✅ Gemma model - capable assistant"
            
        elif "phi" in model_id_lower:
            base_rating = 6
            justification = "✅ Phi model - lightweight but capable"
                
        elif "qwen" in model_id_lower:
            if is_system_related:
                base_rating = 8
                justification = "✅ Qwen model - good system capabilities"
            else:
                base_rating = 7
                justification = "✅ Qwen model - capable assistant"
                
        elif "dolphin" in model_id_lower:
            base_rating = 8
            justification = "✅ Dolphin model - helpful and practical"
                
        elif "hermes" in model_id_lower:
            base_rating = 8
            justification = "✅ Hermes model - strong instruction following"
        
        else:
            base_rating = 6
            justification = "✅ Conversational model"
        
        adjusted_rating = base_rating + size_bonus - size_penalty + practicality_bonus - practicality_penalty
        final_rating = adjusted_rating - crash_penalty
        final_rating = max(1, min(10, final_rating))

        if size_bonus > 0:
            justification += f" | 🚀 {model_size} model"
        elif size_penalty > 0:
            justification += f" | ⚠️ {model_size} model"
        else:
            justification += f" | 📊 {model_size} model"
            
        if practicality_bonus > 0:
            justification += " | 🛠️ Practical helper"
            
        justification += crash_justification
            
        return final_rating, justification

    def get_top_models(self, evaluations: list, top_k: int = 3):
        return evaluations[:top_k]