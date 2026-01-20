# Different attention patterns create different "thinking styles"
class AttentionVariant(Enum):
    LLAMA_ROPE = "rotary_position_embedding"           # Better for long sequences
    GEMMA_GQA = "grouped_query_attention"              # Memory efficient
    PHI_MHA = "multi_head_attention"                   # Standard but effective
    MISTRAL_SLIDING = "sliding_window_attention"       # Local context focus