class Config:
    def __init__(self, vocab_size, context_length, num_layers, d_model, num_heads, d_ff, glu=False):
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.context_length = context_length
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.glu = glu


def resource_accounting(config):
    d_model = config.d_model
    vocab_size = config.vocab_size
    num_layers = config.num_layers
    context_length = config.context_length
    num_heads = config.num_heads
    d_ff = config.d_ff
    glu = config.glu
    """Shared embedding and final lmhead"""

    embedding_params = vocab_size * d_model  # covers both embedding and final lm_head due to weight sharing
    ffn_params = num_layers * ((3 * d_ff * d_model) if glu == True else (2 * d_ff * d_model))
    mha_params = num_layers * (4 * num_heads * d_model * d_model / num_heads)
    ln_params = num_layers * (2 * d_model * 2)  #
    total_parameters = embedding_params + ffn_params + mha_params + ln_params
    print(f"{total_parameters / (1000 * 1000)=}")

    precision = 32 / 8 # single precision float / 8 bits per byte
    parameter_memory = total_parameters * precision
    KQV_memory = context_length * d_model * precision
    memory = (parameter_memory + KQV_memory) / (1024 * 1024)
    print(f"{memory=} GB")

    rough_forward_flops_estimate = 2 * d_model * d_ff * num_layers / (1000 * 1000)
    print(f"{rough_forward_flops_estimate=} million parameters")

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
    resource_accounting(gpt2s_cfg)
