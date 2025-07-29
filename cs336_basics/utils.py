class Config:
    def __init__(self, vocab_size, context_length, num_layers, d_model, num_heads, d_ff, glu=False):
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.context_length = context_length
        self.num_heads = num_heads
        self.d_ff = d_ff


def resource_accounting(config):
    d_model = config.d_model
    vocab_size = config.vocab_size
    num_layers = config.num_layers
    context_length = config.context_length
    num_heads = config.num_heads
    d_ff = config.d_ff

    sequence_length = context_length # not neccessarily the case.
    batch_size = 8

    # Parameters
    positional_params = context_length * d_model
    embedding_params = vocab_size * d_model  # covers both embedding and final lm_head due to weight sharing
    ffn_params = num_layers * (2 * d_ff * d_model)
    mha_params = num_layers * (4 * num_heads * d_model * d_model / num_heads)
    ln_params = num_layers * (2 * d_model * 2)  + d_model * 2 #
    total_parameters = positional_params + embedding_params + ffn_params + mha_params + ln_params
    print(f"{total_parameters / (1000 * 1000)=}")


    # MEMORY

    num_gradients = total_parameters
    optimizer_states = total_parameters * 2  # momentum + variance

    # Activations
    embedding_activations = sequence_length * d_model 
    #ffn
    
    linear_activations =  (d_model + d_ff ) * sequence_length
    # mha

    KQV_activations = sequence_length * d_model * 3
    attention_matrix = sequence_length * sequence_length
    attention_output = sequence_length * d_model
    attention_activations = attention_output + attention_matrix + KQV_activations
    # total activations
    num_activations = (embedding_activations + num_layers * (linear_activations + attention_activations)) * batch_size

    # Total memory
    element_size = 4 # single precision float is 4 bytes 32 / 8
    memory = (num_activations + total_parameters + num_gradients + optimizer_states) * element_size 
    print(f"{memory/1024**3=} GB")

    rough_forward_flops_estimate = (2 * 1 * total_parameters)  # 2 * tokens * num parameters (1 token for one forward pass)
    print(f"{rough_forward_flops_estimate=} FLOPs")

    #
    flops_per_part = {
        "FFN": 0,
        "MHA": 0,
    }
    return total_parameters, flops_per_part


if __name__ == "__main__":
    gpt2xl_cfg = Config(vocab_size=50257, context_length=1024, num_layers=48, d_model=1600, num_heads=25, d_ff=6400)
    gpt2l_cfg = Config(vocab_size=50257, context_length=1024, num_layers=36, d_model=1280, num_heads=20, d_ff=1280 * 4)
    gpt2m_cfg = Config(vocab_size=50257, context_length=1024, num_layers=24, d_model=1024, num_heads=16, d_ff=1024 * 4)
    gpt2s_cfg = Config(vocab_size=50257, context_length=1024, num_layers=12, d_model=768, num_heads=12, d_ff=768 * 4)

    # increased context length
    gpt2xlxc_cfg = Config(vocab_size=50257, context_length=16384, num_layers=48, d_model=1600, num_heads=25, d_ff=6400)
    resource_accounting(gpt2xl_cfg)
