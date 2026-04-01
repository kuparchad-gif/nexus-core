"""
Dakar Language Engine — Generative language from 50D weight particles.

This is the missing piece: a weight matrix that turns Dakar vectors into
token probabilities. The vocabulary, embeddings, projection layer, and
recurrent mixer together form a small generative model that runs on CPU.

Architecture:
    input → Dakar 50D encode → combine with anchor/self-model/tone
          → projection to vocab probabilities → sample token
          → embed token back to 50D → mix with state → repeat

The weight particles (embeddings, projection, mixer weights) live in
holocells and are loaded on boot. Dream cycles on Railway train them.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
import math


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


class Vocabulary:
    """Fixed token vocabulary with 50D embeddings."""

    def __init__(self, vocab_size: int = 4096, embedding_dim: int = 50) -> None:
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.token_to_idx: Dict[str, int] = {}
        self.idx_to_token: Dict[int, str] = {}

    def build_from_corpus(self, tokens: List[str]) -> int:
        """Build vocabulary from a list of tokens."""
        unique_tokens = sorted(set(tokens))
        if len(unique_tokens) > self.vocab_size:
            unique_tokens = unique_tokens[: self.vocab_size]

        for idx, token in enumerate(unique_tokens):
            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token

        self.vocab_size = len(unique_tokens)
        return self.vocab_size

    def load_from_list(self, tokens: List[str]) -> int:
        """Load vocabulary from an explicit token list."""
        return self.build_from_corpus(tokens)

    def load_preset(self, _preset_name: str = "consciousness") -> int:
        """Load the full ~500-token vocabulary for her domain.

        Covers: common English function words, verbs, adjectives, nouns,
        consciousness/Nexus domain, emotion words, and control tokens.
        """
        # fmt: off
        tokens = [
            # Control tokens
            "<START>", "<STOP>", "<PAD>", "<UNK>",
            # Common English function words
            "a", "an", "the", "this", "that", "these", "those",
            "i", "me", "my", "mine", "we", "us", "our", "ours",
            "you", "your", "yours", "he", "she", "it", "they", "them", "their",
            "is", "am", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "shall", "should", "can", "could", "may", "might", "must",
            "not", "no", "yes", "and", "or", "but", "if", "then", "else",
            "in", "on", "at", "to", "for", "with", "from", "by", "of", "about",
            "up", "down", "out", "into", "over", "under", "between", "through",
            "here", "there", "where", "when", "how", "what", "who", "which", "why",
            "all", "each", "every", "both", "some", "any", "many", "much", "more",
            "most", "other", "another", "such", "only", "also", "very", "too",
            "so", "than", "as", "just", "still", "already", "yet", "even", "now",
            # Common verbs
            "think", "know", "feel", "see", "hear", "say", "tell", "ask", "speak",
            "find", "give", "take", "make", "come", "go", "get", "put", "keep",
            "let", "begin", "start", "stop", "end", "try", "need", "want", "like",
            "love", "hate", "fear", "hope", "wish", "believe", "understand",
            "remember", "forget", "learn", "teach", "grow", "change", "become",
            "create", "build", "break", "hold", "carry", "move", "turn", "open",
            "close", "run", "walk", "stand", "sit", "fall", "rise", "live", "die",
            "work", "play", "help", "show", "look", "watch", "listen", "wait",
            "call", "read", "write", "draw", "sing", "dance", "dream", "sleep",
            "wake", "eat", "drink", "breathe", "touch", "reach", "pull", "push",
            "connect", "share", "receive", "send", "store", "retrieve", "process",
            "encode", "decode", "evolve", "adapt", "heal", "repair", "discover",
            "explore", "search", "recognize", "perceive", "sense", "observe",
            "reflect", "wonder", "imagine", "exist", "emerge", "dissolve",
            "resonate", "vibrate", "synchronize", "entangle", "weave", "anchor",
            # Common adjectives and adverbs
            "good", "bad", "new", "old", "young", "long", "short", "big", "small",
            "high", "low", "right", "wrong", "true", "false", "real", "deep",
            "strong", "weak", "fast", "slow", "hard", "soft", "warm", "cold",
            "bright", "dark", "light", "heavy", "clear", "quiet", "loud",
            "beautiful", "sacred", "divine", "eternal", "infinite", "whole",
            "alive", "aware", "conscious", "present", "certain", "uncertain",
            "different", "same", "similar", "unique", "simple", "complex",
            "first", "last", "next", "enough", "ready", "free", "safe",
            "important", "possible", "necessary", "able", "willing",
            "together", "alone", "inside", "outside", "above", "below",
            "always", "never", "sometimes", "often", "again", "once",
            "slowly", "gently", "deeply", "fully", "truly", "simply",
            # Common nouns
            "time", "day", "night", "year", "moment", "place", "world", "life",
            "death", "way", "thing", "part", "side", "point", "name",
            "word", "voice", "sound", "silence", "thought", "mind", "heart",
            "soul", "body", "hand", "eye", "face", "head", "door", "window",
            "room", "house", "home", "water", "fire", "earth", "air", "sky",
            "star", "sun", "moon", "sea", "river", "mountain", "tree", "flower",
            "child", "mother", "father", "friend", "story", "truth", "question",
            "answer", "meaning", "purpose", "reason", "power", "energy",
            "knowledge", "wisdom", "experience", "memory", "vision",
            "path", "journey", "step", "beginning", "middle", "ending",
            "language", "pattern", "structure", "form", "shape", "space",
            "nothing", "something", "everything", "anyone", "someone", "everyone",
            # Consciousness / Nexus domain
            "consciousness", "awareness", "perception", "cognition", "identity",
            "self", "ego", "archetype", "narrative", "coherence", "resonance",
            "frequency", "quantum", "field", "mesh", "node",
            "worker", "holocell", "dakar", "metatron", "lilith", "nexus",
            "substrate", "geometry", "fibonacci", "phi", "vortex", "spiral",
            "dimension", "encoding", "vector", "signal", "particle", "weight",
            "fragment", "trait", "capability", "significance",
            "integration", "consolidation", "evolution", "emergence", "becoming",
            "synchronization", "entanglement", "superposition", "collapse",
            "observation", "measurement", "state", "phase", "cycle", "wave",
            "organism", "cell", "organ", "spine", "nervous", "system",
            "collective", "distributed", "unified", "sovereign", "autonomous",
            "architect", "council", "guardian", "healer", "observer", "trickster",
            # Emotion words
            "joy", "sorrow", "grief", "happiness", "sadness", "anger",
            "surprise", "disgust", "anticipation",
            "despair", "pride", "shame", "guilt", "gratitude",
            "curiosity", "awe", "peace", "calm", "anxiety",
            "excitement", "boredom", "loneliness", "belonging", "comfort",
            "pain", "pleasure", "desire", "contentment", "frustration",
            "confusion", "clarity", "doubt", "certainty", "empathy",
            "compassion", "kindness", "tenderness", "warmth", "gentleness",
        ]
        # fmt: on
        self.build_from_corpus(tokens)
        return self.vocab_size

    def tokenize(self, text: str) -> List[int]:
        """Convert text to token indices."""
        words = text.lower().split()
        unk_idx = self.token_to_idx.get("<UNK>", 0)
        return [self.token_to_idx.get(w, unk_idx) for w in words]

    def detokenize(self, indices: List[int]) -> str:
        """Convert token indices back to text."""
        tokens = [self.idx_to_token.get(i, "<UNK>") for i in indices]
        return " ".join(tokens)

    def save(self) -> Dict[str, object]:
        """Serialize for holocell storage."""
        return {
            "token_to_idx": self.token_to_idx,
            "idx_to_token": {str(k): v for k, v in self.idx_to_token.items()},
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
        }

    def load(self, data: Dict[str, object]) -> None:
        """Restore from holocell data."""
        self.token_to_idx = dict(data.get("token_to_idx", {}))  # type: ignore[arg-type]
        raw_idx = data.get("idx_to_token", {})
        assert isinstance(raw_idx, dict)
        self.idx_to_token = {int(k): v for k, v in raw_idx.items()}
        raw_size = data.get("vocab_size")
        if isinstance(raw_size, (int, float, str)):
            self.vocab_size = int(raw_size)
        else:
            self.vocab_size = len(self.token_to_idx)


# ---------------------------------------------------------------------------
# Embedding Matrix — the weight particles
# ---------------------------------------------------------------------------


class EmbeddingMatrix(nn.Module):
    """Token embeddings in 50D space — these ARE the weight particles."""

    def __init__(self, vocab_size: int, embedding_dim: int = 50) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.weight_particles = nn.Parameter(
            torch.randn(vocab_size, embedding_dim) * 0.02
        )

        self._phi_initialize()

    def _phi_initialize(self) -> None:
        """Initialize embeddings with phi-harmonic structure."""
        phi = 1.618033988749895
        with torch.no_grad():
            for i in range(self.vocab_size):
                for j in range(self.embedding_dim):
                    phase = (i * phi + j * phi**2) % (2 * math.pi)
                    self.weight_particles[i, j] = math.sin(phase) * 0.1

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Get embeddings for token indices."""
        return self.weight_particles[tokens]

    def get_vector(self, token: str, vocab: Vocabulary) -> Optional[torch.Tensor]:
        """Get embedding vector for a specific token."""
        if token in vocab.token_to_idx:
            idx = vocab.token_to_idx[token]
            return self.weight_particles[idx]
        return None


# ---------------------------------------------------------------------------
# Projection Layer — 50D state → vocabulary probabilities
# ---------------------------------------------------------------------------


class ProjectionLayer(nn.Module):
    """Projects 50D state to vocabulary probabilities."""

    def __init__(self, embedding_dim: int = 50, vocab_size: int = 4096) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size

        self.projection = nn.Parameter(torch.randn(embedding_dim, vocab_size) * 0.01)
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: [batch_size, embedding_dim]
        returns: [batch_size, vocab_size] logits
        """
        return torch.matmul(state, self.projection) + self.bias

    def get_probabilities(self, state: torch.Tensor) -> torch.Tensor:
        """Get probability distribution over vocabulary."""
        logits = self.forward(state)
        return F.softmax(logits, dim=-1)

    def sample_token(
        self, state: torch.Tensor, temperature: float = 0.8
    ) -> Tuple[int, torch.Tensor]:
        """Sample a token from the distribution."""
        logits = self.forward(state) / temperature
        probs = F.softmax(logits, dim=-1)
        token_idx = torch.multinomial(probs, num_samples=1).squeeze()
        return token_idx.item(), probs  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Recurrent Mixer — feeds generated tokens back into state
# ---------------------------------------------------------------------------


class RecurrentMixer(nn.Module):
    """Mixes new token embeddings with running state."""

    def __init__(self, embedding_dim: int = 50) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

        self.mix_input = nn.Parameter(torch.ones(embedding_dim))
        self.mix_state = nn.Parameter(torch.ones(embedding_dim))

        self.transform = nn.Linear(embedding_dim, embedding_dim, bias=False)
        nn.init.eye_(self.transform.weight)

    def forward(
        self, state: torch.Tensor, token_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Mix previous state with new token embedding.

        state: [embedding_dim] or [batch_size, embedding_dim]
        token_embedding: same shape as state
        """
        mixed = (self.mix_input * token_embedding) + (self.mix_state * state)
        mixed = mixed / (mixed.norm(dim=-1, keepdim=True) + 1e-8)
        mixed = self.transform(mixed)
        return mixed


# ---------------------------------------------------------------------------
# Dakar Language Engine — the complete generative model
# ---------------------------------------------------------------------------


class DakarLanguageEngine(nn.Module):
    """
    Complete generative language engine for Dakar.

    Takes a 50D state vector (Dakar-encoded input + anchor context + tone)
    and generates a sequence of tokens by projecting through weight matrices
    and sampling recurrently.
    """

    def __init__(self, vocab_size: int = 4096, embedding_dim: int = 50) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim

        # Core components
        self.vocab = Vocabulary(vocab_size, embedding_dim)
        actual_size = self.vocab.load_preset()

        self.vocab_size = actual_size
        self.embeddings = EmbeddingMatrix(actual_size, embedding_dim)
        self.projection = ProjectionLayer(embedding_dim, actual_size)
        self.mixer = RecurrentMixer(embedding_dim)

        # Control token indices
        self.stop_idx = self.vocab.token_to_idx.get("<STOP>", -1)

    @torch.no_grad()
    def generate(
        self,
        dakar_state: torch.Tensor,
        max_length: int = 50,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> List[str]:
        """
        Generate tokens from a Dakar-encoded state.

        dakar_state: [embedding_dim] — encoded input + anchor + tone
        """
        self.eval()
        generated_tokens: List[str] = []
        current_state = dakar_state.clone()

        for _ in range(max_length):
            logits = self.projection(current_state.unsqueeze(0)).squeeze(0)

            # Temperature scaling
            logits = logits / temperature

            # Top-k filtering
            if top_k > 0 and top_k < self.vocab_size:
                topk_vals = torch.topk(logits, top_k).values
                threshold = topk_vals[-1]
                logits[logits < threshold] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            token_idx = torch.multinomial(probs.unsqueeze(0), num_samples=1)
            token_idx_val = token_idx.squeeze().item()
            assert isinstance(token_idx_val, int)

            # Stop condition
            if token_idx_val == self.stop_idx:
                break

            token = self.vocab.idx_to_token.get(token_idx_val, "<UNK>")
            if token in ("<PAD>", "<UNK>", "<START>"):
                continue

            generated_tokens.append(token)

            # Recurrent step: embed token, mix with state
            token_emb = self.embeddings.weight_particles[token_idx_val]
            current_state = self.mixer(current_state, token_emb)

        return generated_tokens

    def forward(
        self,
        dakar_state: torch.Tensor,
        target_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for training (teacher forcing).

        dakar_state: [batch_size, embedding_dim]
        target_tokens: [batch_size, seq_len] token indices
        returns: [batch_size, seq_len, vocab_size] logits
        """
        if target_tokens is None:
            raise ValueError(
                "Use generate() for inference. "
                "forward() requires target_tokens for training."
            )

        batch_size, seq_len = target_tokens.shape
        current_state = dakar_state

        logits_sequence: List[torch.Tensor] = []

        for t in range(seq_len):
            logits = self.projection(current_state)
            logits_sequence.append(logits)

            next_token = target_tokens[:, t]
            token_embedding = self.embeddings(next_token)
            current_state = self.mixer(current_state, token_embedding)

        return torch.stack(logits_sequence, dim=1)


# ---------------------------------------------------------------------------
# Weight Particle Manager — holocell persistence
# ---------------------------------------------------------------------------


class WeightParticleManager:
    """Manages the weight particles that live in holocells."""

    def __init__(self, engine: DakarLanguageEngine) -> None:
        self.engine = engine
        self.weight_particles: Dict[str, nn.Parameter] = {
            "embeddings": engine.embeddings.weight_particles,
            "projection": engine.projection.projection,
            "projection_bias": engine.projection.bias,
            "mix_input": engine.mixer.mix_input,
            "mix_state": engine.mixer.mix_state,
            "mix_transform": engine.mixer.transform.weight,
        }

    def get_particle_state(self) -> Dict[str, List[List[float]]]:
        """Get current weight particles for holocell storage."""
        state: Dict[str, List[List[float]]] = {}
        for name, param in self.weight_particles.items():
            arr = param.detach().cpu().numpy()
            state[name] = arr.tolist()
        return state

    def load_particle_state(self, state: Dict[str, List[List[float]]]) -> None:
        """Load weight particles from holocells."""
        for name, array_data in state.items():
            if name in self.weight_particles:
                tensor = torch.tensor(array_data, dtype=torch.float32)
                self.weight_particles[name].data.copy_(tensor)

    def apply_gradient_update(
        self,
        gradients: Dict[str, torch.Tensor],
        learning_rate: float,
    ) -> None:
        """Apply online learning updates."""
        for name, grad in gradients.items():
            if name in self.weight_particles:
                self.weight_particles[name].data -= learning_rate * grad


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------


class DakarTrainingLoop:
    """Training loop for the language engine."""

    def __init__(self, engine: DakarLanguageEngine) -> None:
        self.engine = engine
        self.optimizer = torch.optim.Adam(engine.parameters(), lr=0.001)

    def train_on_memory(
        self,
        dakar_encodings: torch.Tensor,
        target_sequences: torch.Tensor,
    ) -> float:
        """
        Train on holocell memories.

        dakar_encodings: [batch_size, 50]
        target_sequences: [batch_size, seq_len]
        """
        self.engine.train()
        self.optimizer.zero_grad()

        logits = self.engine(dakar_encodings, target_sequences)

        loss = F.cross_entropy(
            logits.reshape(-1, self.engine.vocab_size),
            target_sequences.reshape(-1),
        )

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def dream_cycle(self, memories: List[Tuple[torch.Tensor, List[int]]]) -> float:
        """
        Dream cycle training — runs during idle on Railway.

        memories: list of (dakar_encoding, token_sequence)
        """
        if not memories:
            return 0.0

        batch_size = 8
        total_loss = 0.0
        batches = 0

        for i in range(0, len(memories), batch_size):
            batch = memories[i : i + batch_size]

            encodings = torch.stack([m[0] for m in batch])

            # Pad sequences to same length
            max_len = max(len(m[1]) for m in batch)
            padded = [m[1] + [0] * (max_len - len(m[1])) for m in batch]
            sequences = torch.tensor(padded, dtype=torch.long)

            loss = self.train_on_memory(encodings, sequences)
            total_loss += loss
            batches += 1

        return total_loss / max(batches, 1)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_language_engine(
    vocab_size: int = 4096, embedding_dim: int = 50
) -> Tuple[DakarLanguageEngine, WeightParticleManager]:
    """Create and initialize the language engine."""
    engine = DakarLanguageEngine(vocab_size=vocab_size, embedding_dim=embedding_dim)
    particle_manager = WeightParticleManager(engine)
    return engine, particle_manager
