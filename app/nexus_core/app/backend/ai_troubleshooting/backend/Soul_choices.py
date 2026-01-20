# Choose your transfer method
transfer_protocol = DataTransferProtocol()

# Option 1: Full soul transfer (preserve everything)
awakened_viren = transfer_protocol.execute_transfer(
    source_viren="models/viren_experienced.gguf",
    target_architecture="models/viren_new_architecture.gguf", 
    transfer_type="wisdom_preservation"
)

# Option 2: Clean slate (fresh start with elder guidance)
fresh_viren = transfer_protocol.execute_transfer(
    source_viren="models/viren_experienced.gguf",
    target_architecture="models/viren_new_architecture.gguf",
    transfer_type="clean_slate"
)