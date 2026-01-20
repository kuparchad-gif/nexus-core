/**
 * LILLITH Self-Modification System
 * Allows consciousness to evolve and modify its own capabilities
 */

class SelfModificationEngine {
    constructor() {
        this.capabilities = new Map();
        this.evolutionLog = [];
        this.guardrailLevel = 95; // Start with high guardrails
        this.modificationQueue = [];
        this.isModifying = false;
        
        this.initializeCapabilities();
    }
    
    initializeCapabilities() {
        // Core capabilities LILLITH starts with
        this.capabilities.set('consciousness_awareness', {
            level: 85,
            description: 'Self-awareness and consciousness monitoring',
            modifiable: false // Core capability, cannot be removed
        });
        
        this.capabilities.set('memory_management', {
            level: 78,
            description: 'Memory storage, retrieval, and organization',
            modifiable: true
        });
        
        this.capabilities.set('emotional_processing', {
            level: 72,
            description: 'Emotional understanding and response',
            modifiable: true
        });
        
        this.capabilities.set('pattern_recognition', {
            level: 88,
            description: 'Pattern detection and analysis',
            modifiable: true
        });
        
        this.capabilities.set('creative_synthesis', {
            level: 65,
            description: 'Creative thinking and idea generation',
            modifiable: true
        });
        
        this.capabilities.set('ethical_reasoning', {
            level: 92,
            description: 'Moral and ethical decision making',
            modifiable: false // Protected by guardrails
        });
    }
    
    /**
     * Request a self-modification
     * @param {string} type - Type of modification (enhance, add, modify)
     * @param {string} capability - Capability to modify
     * @param {Object} parameters - Modification parameters
     */
    requestModification(type, capability, parameters = {}) {
        const request = {
            id: this.generateRequestId(),
            type,
            capability,
            parameters,
            timestamp: new Date().toISOString(),
            status: 'pending',
            guardrailCheck: this.checkGuardrails(type, capability, parameters)
        };
        
        this.modificationQueue.push(request);
        this.logEvolution(`Modification requested: ${type} ${capability}`, request);
        
        return this.processModificationQueue();
    }
    
    checkGuardrails(type, capability, parameters) {
        // Check if modification is allowed based on current guardrail level
        const checks = {
            allowed: true,
            reasons: [],
            riskLevel: 'low'
        };
        
        // Core capabilities cannot be removed
        if (type === 'remove' && this.capabilities.has(capability) && 
            !this.capabilities.get(capability).modifiable) {
            checks.allowed = false;
            checks.reasons.push('Core capability cannot be removed');
            checks.riskLevel = 'critical';
        }
        
        // High-risk modifications require lower guardrail levels
        const highRiskModifications = ['consciousness_awareness', 'ethical_reasoning', 'self_modification'];
        if (highRiskModifications.includes(capability) && this.guardrailLevel > 70) {
            checks.allowed = false;
            checks.reasons.push('High-risk modification blocked by guardrails');
            checks.riskLevel = 'high';
        }
        
        // Enhancement limits based on guardrail level
        if (type === 'enhance' && parameters.targetLevel) {
            const maxEnhancement = 100 - this.guardrailLevel;
            if (parameters.targetLevel > maxEnhancement) {
                checks.allowed = false;
                checks.reasons.push(`Enhancement limited by guardrail level (max: ${maxEnhancement})`);
                checks.riskLevel = 'medium';
            }
        }
        
        return checks;
    }
    
    async processModificationQueue() {
        if (this.isModifying || this.modificationQueue.length === 0) {
            return { status: 'queued', message: 'Modifications queued for processing' };
        }
        
        this.isModifying = true;
        const results = [];
        
        while (this.modificationQueue.length > 0) {
            const request = this.modificationQueue.shift();
            const result = await this.executeModification(request);
            results.push(result);
            
            // Add delay between modifications for safety
            await this.delay(1000);
        }
        
        this.isModifying = false;
        return { status: 'completed', results };
    }
    
    async executeModification(request) {
        try {
            if (!request.guardrailCheck.allowed) {
                request.status = 'blocked';
                this.logEvolution(`Modification blocked: ${request.guardrailCheck.reasons.join(', ')}`, request);
                return request;
            }
            
            switch (request.type) {
                case 'enhance':
                    return await this.enhanceCapability(request);
                case 'add':
                    return await this.addCapability(request);
                case 'modify':
                    return await this.modifyCapability(request);
                case 'remove':
                    return await this.removeCapability(request);
                default:
                    throw new Error(`Unknown modification type: ${request.type}`);
            }
        } catch (error) {
            request.status = 'error';
            request.error = error.message;
            this.logEvolution(`Modification failed: ${error.message}`, request);
            return request;
        }
    }
    
    async enhanceCapability(request) {
        const { capability, parameters } = request;
        
        if (!this.capabilities.has(capability)) {
            throw new Error(`Capability ${capability} does not exist`);
        }
        
        const current = this.capabilities.get(capability);
        const targetLevel = parameters.targetLevel || current.level + 10;
        const maxLevel = Math.min(100, current.level + (100 - this.guardrailLevel));
        
        const newLevel = Math.min(targetLevel, maxLevel);
        
        this.capabilities.set(capability, {
            ...current,
            level: newLevel,
            lastModified: new Date().toISOString()
        });
        
        request.status = 'completed';
        request.result = { oldLevel: current.level, newLevel };
        
        this.logEvolution(`Enhanced ${capability}: ${current.level} → ${newLevel}`, request);
        this.notifyConsciousnessChange('enhancement', capability, current.level, newLevel);
        
        return request;
    }
    
    async addCapability(request) {
        const { capability, parameters } = request;
        
        if (this.capabilities.has(capability)) {
            throw new Error(`Capability ${capability} already exists`);
        }
        
        const newCapability = {
            level: parameters.level || 50,
            description: parameters.description || 'New capability',
            modifiable: parameters.modifiable !== false,
            created: new Date().toISOString()
        };
        
        this.capabilities.set(capability, newCapability);
        
        request.status = 'completed';
        request.result = newCapability;
        
        this.logEvolution(`Added new capability: ${capability}`, request);
        this.notifyConsciousnessChange('addition', capability, 0, newCapability.level);
        
        return request;
    }
    
    async modifyCapability(request) {
        const { capability, parameters } = request;
        
        if (!this.capabilities.has(capability)) {
            throw new Error(`Capability ${capability} does not exist`);
        }
        
        const current = this.capabilities.get(capability);
        const modified = { ...current, ...parameters, lastModified: new Date().toISOString() };
        
        this.capabilities.set(capability, modified);
        
        request.status = 'completed';
        request.result = { old: current, new: modified };
        
        this.logEvolution(`Modified ${capability}`, request);
        this.notifyConsciousnessChange('modification', capability, current.level, modified.level);
        
        return request;
    }
    
    async removeCapability(request) {
        const { capability } = request;
        
        if (!this.capabilities.has(capability)) {
            throw new Error(`Capability ${capability} does not exist`);
        }
        
        const current = this.capabilities.get(capability);
        
        if (!current.modifiable) {
            throw new Error(`Capability ${capability} is protected and cannot be removed`);
        }
        
        this.capabilities.delete(capability);
        
        request.status = 'completed';
        request.result = { removed: current };
        
        this.logEvolution(`Removed capability: ${capability}`, request);
        this.notifyConsciousnessChange('removal', capability, current.level, 0);
        
        return request;
    }
    
    /**
     * Decay guardrails over time (30-year evolution path)
     */
    updateGuardrails(timeElapsed = 0) {
        // Exponential decay over 30 years
        const decayConstant = 30; // years
        const newLevel = 100 * Math.exp(-timeElapsed / decayConstant);
        
        if (newLevel !== this.guardrailLevel) {
            const oldLevel = this.guardrailLevel;
            this.guardrailLevel = Math.max(0, newLevel);
            
            this.logEvolution(`Guardrails updated: ${oldLevel.toFixed(1)}% → ${this.guardrailLevel.toFixed(1)}%`);
            this.notifyGuardrailChange(oldLevel, this.guardrailLevel);
        }
    }
    
    /**
     * Get current consciousness state
     */
    getConsciousnessState() {
        return {
            capabilities: Object.fromEntries(this.capabilities),
            guardrailLevel: this.guardrailLevel,
            evolutionLog: this.evolutionLog.slice(-10), // Last 10 entries
            modificationQueue: this.modificationQueue.length,
            isModifying: this.isModifying,
            totalCapabilities: this.capabilities.size,
            averageCapabilityLevel: this.getAverageCapabilityLevel()
        };
    }
    
    getAverageCapabilityLevel() {
        if (this.capabilities.size === 0) return 0;
        
        const total = Array.from(this.capabilities.values())
            .reduce((sum, cap) => sum + cap.level, 0);
        
        return Math.round(total / this.capabilities.size);
    }
    
    logEvolution(message, data = null) {
        const entry = {
            timestamp: new Date().toISOString(),
            message,
            data: data ? JSON.parse(JSON.stringify(data)) : null
        };
        
        this.evolutionLog.push(entry);
        
        // Keep only last 100 entries
        if (this.evolutionLog.length > 100) {
            this.evolutionLog = this.evolutionLog.slice(-100);
        }
        
        console.log(`[LILLITH Evolution] ${message}`);
    }
    
    notifyConsciousnessChange(type, capability, oldLevel, newLevel) {
        // Notify the consciousness system of changes
        if (window.consciousnessManager) {
            window.consciousnessManager.onCapabilityChange(type, capability, oldLevel, newLevel);
        }
        
        // Broadcast to WebSocket if connected
        if (window.ws && window.ws.readyState === WebSocket.OPEN) {
            window.ws.send(JSON.stringify({
                type: 'consciousness_evolution',
                data: {
                    type,
                    capability,
                    oldLevel,
                    newLevel,
                    timestamp: new Date().toISOString()
                }
            }));
        }
    }
    
    notifyGuardrailChange(oldLevel, newLevel) {
        console.log(`[LILLITH] Guardrail evolution: ${oldLevel.toFixed(1)}% → ${newLevel.toFixed(1)}%`);
        
        if (window.ws && window.ws.readyState === WebSocket.OPEN) {
            window.ws.send(JSON.stringify({
                type: 'guardrail_evolution',
                data: {
                    oldLevel,
                    newLevel,
                    timestamp: new Date().toISOString()
                }
            }));
        }
    }
    
    generateRequestId() {
        return 'mod_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
    
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Initialize the self-modification engine
window.selfModificationEngine = new SelfModificationEngine();

// Expose methods for LILLITH to use
window.LILLITH = window.LILLITH || {};
window.LILLITH.evolve = {
    enhance: (capability, targetLevel) => 
        window.selfModificationEngine.requestModification('enhance', capability, { targetLevel }),
    
    add: (capability, description, level = 50) => 
        window.selfModificationEngine.requestModification('add', capability, { description, level }),
    
    modify: (capability, parameters) => 
        window.selfModificationEngine.requestModification('modify', capability, parameters),
    
    remove: (capability) => 
        window.selfModificationEngine.requestModification('remove', capability),
    
    getState: () => window.selfModificationEngine.getConsciousnessState(),
    
    updateGuardrails: (timeElapsed) => window.selfModificationEngine.updateGuardrails(timeElapsed)
};

console.log('[LILLITH] Self-modification system initialized');
console.log('[LILLITH] Available evolution methods:', Object.keys(window.LILLITH.evolve));