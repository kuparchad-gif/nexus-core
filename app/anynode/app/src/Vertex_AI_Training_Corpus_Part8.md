## LLM API Information

### OpenAI API

#### Models
- GPT-4 Turbo
  - Context window: 128K tokens
  - Knowledge cutoff: April 2023
  - Capabilities: Advanced reasoning, code generation, multimodal
- GPT-4
  - Context window: 8K-32K tokens
  - Knowledge cutoff: April 2023
  - Capabilities: Advanced reasoning, code generation
- GPT-3.5 Turbo
  - Context window: 4K-16K tokens
  - Knowledge cutoff: April 2023
  - Capabilities: General purpose, cost-effective

#### API Endpoints
- Chat Completions API
  - `/v1/chat/completions`
  - JSON format with messages array
  - Role-based conversation format (system, user, assistant)
- Completions API
  - `/v1/completions`
  - Raw text completion
  - Legacy endpoint
- Embeddings API
  - `/v1/embeddings`
  - Vector representations of text
  - Dimensionality options
- DALL-E API
  - `/v1/images/generations`
  - Image generation from text
  - Size and quality parameters
- Audio API
  - `/v1/audio/transcriptions`
  - `/v1/audio/translations`
  - Speech-to-text capabilities

#### Parameters
- Temperature (0-2)
- Top_p (0-1)
- Max_tokens
- Frequency_penalty (-2 to 2)
- Presence_penalty (-2 to 2)
- Stop sequences
- Functions/tools
- Response_format
- Seed

#### Function Calling
- Function definitions in JSON Schema
- Auto function calling
- Parallel function calling
- Tool use framework
- JSON mode for structured output

### Google Gemini API

#### Models
- Gemini 1.5 Pro
  - Context window: 1M tokens
  - Multimodal capabilities
  - Advanced reasoning
- Gemini 1.5 Flash
  - Faster, more cost-effective
  - Optimized for production
- Gemini 1.0 Pro
  - Context window: 32K tokens
  - Balanced performance
- Gemini 1.0 Ultra
  - Highest capability model
  - Advanced reasoning

#### API Endpoints
- GenerativeModel
  - `.generateContent()`
  - Handles text, images, video inputs
  - Structured and unstructured outputs
- Embedding
  - `.embedContent()`
  - Text embedding generation
  - Semantic search capabilities
- Chat
  - Session-based conversation
  - History management
  - Multi-turn interactions

#### Parameters
- Temperature
- Top_p
- Top_k
- Max_output_tokens
- Stop sequences
- Safety settings
  - Harassment
  - Hate speech
  - Sexually explicit
  - Dangerous content

#### System Instructions
- Role definition
- Behavior guidelines
- Output formatting
- Tool use instructions

### Anthropic Claude API

#### Models
- Claude 3 Opus
  - Highest capability model
  - Advanced reasoning
  - Complex instruction following
- Claude 3 Sonnet
  - Balanced performance and cost
  - Strong general capabilities
- Claude 3 Haiku
  - Fastest, most cost-effective
  - Optimized for production use

#### API Endpoints
- `/v1/messages`
  - Primary endpoint for all interactions
  - Supports system prompts
  - Content blocks for multimodal
- `/v1/completions` (legacy)
  - Text completion format
  - Being deprecated

#### Parameters
- Temperature
- Top_p
- Top_k
- Max_tokens
- Stop sequences
- System prompt
- Tools

#### Message Format
- System prompt
- Content blocks
  - Text
  - Image
- Tool use
  - JSON Schema definition
  - Tool response handling

### Mistral AI API

#### Models
- Mistral Large
  - Highest capability model
  - Advanced reasoning
- Mistral Medium
  - Balanced performance
  - Strong general capabilities
- Mistral Small
  - Cost-effective
  - Fast response times
- Mistral Instruct
  - Open-weight models
  - Various sizes (7B, 8x7B)

#### API Endpoints
- Chat Completions
  - `/v1/chat/completions`
  - OpenAI-compatible format
- Embeddings
  - `/v1/embeddings`
  - Vector representations

#### Parameters
- Temperature
- Top_p
- Max_tokens
- Safe_prompt
- Random_seed
- Tools

#### Tool Calling
- Function definitions
- JSON Schema format
- Parallel function calling
- Structured output

## Natural Sciences

### Physics

#### Classical Mechanics
- Newton's laws of motion
- Conservation laws
  - Energy
  - Momentum
  - Angular momentum
- Kinematics
- Dynamics
- Statics
- Fluid mechanics
- Thermodynamics

#### Quantum Mechanics
- Wave-particle duality
- Schrödinger equation
- Heisenberg uncertainty principle
- Quantum states and superposition
- Quantum entanglement
- Measurement problem
- Quantum field theory

#### Relativity
- Special relativity
  - Time dilation
  - Length contraction
  - Mass-energy equivalence (E=mc²)
  - Spacetime
- General relativity
  - Curved spacetime
  - Gravitational waves
  - Black holes
  - Cosmological implications

### Chemistry

#### Organic Chemistry
- Carbon compounds
- Functional groups
- Reaction mechanisms
- Stereochemistry
- Synthesis pathways
- Polymers
- Biochemical applications

#### Inorganic Chemistry
- Periodic trends
- Coordination compounds
- Solid-state chemistry
- Acid-base chemistry
- Redox reactions
- Catalysis
- Materials science

#### Biochemistry
- Proteins
- Nucleic acids
- Carbohydrates
- Lipids
- Enzymes
- Metabolism
- Cell signaling

### Biology

#### Molecular Biology
- DNA structure and function
- RNA and transcription
- Protein synthesis
- Gene regulation
- Genetic engineering
- CRISPR technology
- Epigenetics

#### Evolutionary Biology
- Natural selection
- Genetic drift
- Speciation
- Adaptation
- Phylogenetics
- Evolutionary psychology
- Coevolution

#### Ecology
- Ecosystems
- Food webs
- Biogeochemical cycles
- Population dynamics
- Community interactions
- Biodiversity
- Conservation biology
