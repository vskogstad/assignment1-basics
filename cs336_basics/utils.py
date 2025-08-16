class ResourceConfig:
    def __init__(self, vocab_size, context_length, num_layers, d_model, num_heads, d_ff, batch_size=1024, model_name = "transformer_silu", theta= None):
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.context_length = context_length
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.model_name = model_name
        self.theta = theta
        self.batch_size = batch_size


def resource_accounting(config):
    """Not happy with this. FLOPS/memory is in the right ball park/wrong respectively. I think parameters are correct."""
    d_model = config.d_model
    vocab_size = config.vocab_size
    num_layers = config.num_layers
    context_length = config.context_length
    num_heads = config.num_heads
    d_ff = config.d_ff
    batch_size = config.batch_size
    model_arc = config.model_name
    theta = config.theta
    sequence_length = context_length  # not neccessarily the case.
    weight_sharing = False if config.theta else True
    num_weight_matrices = 2 if model_arc == "tranformer_silu" else 3 # We run GLU for all other variants


    # Parameters
    positional_params = context_length * d_model if not theta else 0 # If RoPE, no positional params
    embedding_params = vocab_size * d_model  # covers both embedding and final lm_head. Only one matrix due to weight sharing for gpt-2
    lm_head_params = vocab_size * d_model * (1 - weight_sharing)
    ffn_params = num_layers * (num_weight_matrices * d_ff * d_model)  # 3 matrices with GLU, 2 with just LU
    mha_params = num_layers * (4 * num_heads * d_model * d_model / num_heads)  # 4 = K,Q,V,O
    ln_params = num_layers * (2 * d_model * 2) + d_model * 2  #
    non_embedding_params = positional_params + ffn_params + mha_params + ln_params + lm_head_params
    total_parameters = non_embedding_params + embedding_params
    print(f"Total parameters = {total_parameters / (1000 * 1000 ):.2f} M")

    # FLOPS (skipped rmsnorm)
    pos_flops = 1 * sequence_length * d_model
    ffn_flops = num_layers * (4 * sequence_length * d_ff * d_model)
    mha_flops = num_layers * (
        (3 * 2 * sequence_length * d_model * d_model)  # Q K and V
        + (2 * sequence_length * d_model * sequence_length)  # Q @ K^T
        + (2 * sequence_length * sequence_length * d_model)  # attn @ V
        + (2 * sequence_length * d_model * d_model)  # out @ W_O
    )
    lmhead_flops = 2 * (sequence_length * vocab_size * d_model)
    total_flops = pos_flops + ffn_flops + mha_flops + lmhead_flops
    flops_per_part = {"Pos flops": pos_flops/total_flops, 
                      "FFN flops": ffn_flops/total_flops, 
                      "Attn flops": mha_flops/total_flops, 
                      "Lmhead flops": lmhead_flops/total_flops}
    print(f"Forward flops = {total_flops/1e12:.2f} TFLOPs")
    flops_per_batch = total_flops * 3 * batch_size
    flops_train = 400_000 * 1024 * 3 * total_flops/1e12
    #print(f"Total flops A100, 400k steps, 1024 batch_size= {flops_train:.2f} TFLOPs")
    #print(f"Total time = {flops_train / (19.5 * 0.5 * 3600 * 24):.2f} days")
    [print(f"{k} = {v*100:.2f}%")for k,v in flops_per_part.items()]

    # MEMORY
    num_gradients = total_parameters
    optimizer_states = total_parameters * 2  # momentum + variance

    # Activations
    embedding_activations = sequence_length * d_model
    # ffn

    linear_activations = (d_model + d_ff) * sequence_length
    # mha

    KQV_activations = sequence_length * d_model * 3
    attention_matrix = sequence_length * sequence_length
    attention_output = sequence_length * d_model
    attention_activations = attention_output + attention_matrix + KQV_activations
    # total activations
    num_activations = (embedding_activations + num_layers * (linear_activations + attention_activations)) * batch_size

    # Total memory
    element_size = 4  # single precision float is 4 bytes 32 / 8
    memory = (num_activations + total_parameters + num_gradients + optimizer_states) * element_size / 1024**3
    print(f"{memory = :.2f} GB")
    # loading model only
    loading_memory = total_parameters * element_size / 1024**3
    print(f"{loading_memory = :.2f} GB")

    rough_forward_flops_estimate = (
        2 * sequence_length * total_parameters
    )  # 2 * tokens * num parameters (1 token for one forward pass)
    #print(f"{rough_forward_flops_estimate/1e12 = :.2f} TFLOPs")

    return flops_per_batch, non_embedding_params  # FLOPs for forward and backward pass per batch

def step_law_lr(len_data, non_embedding_params, context_length):
    opti_batch = 0.58 * len_data ** 0.571
    
    opti_lr = 1.79 * non_embedding_params **(-0.713) * len_data **0.307
    print(f"Dataset is {len_data:,} tokens. \nOptimal batch size = {opti_batch} tokens or {opti_batch/context_length} sequence_batches. \nOptimal lr = {opti_lr:.4f}")

if __name__ == "__main__":
    gpt2xl_cfg = ResourceConfig(vocab_size=50257, context_length=1024, num_layers=48, d_model=1600, num_heads=25, d_ff=6400)
    gpt2l_cfg = ResourceConfig(vocab_size=50257, context_length=1024, num_layers=36, d_model=1280, num_heads=20, d_ff=1280 * 4)
    gpt2m_cfg = ResourceConfig(vocab_size=50257, context_length=1024, num_layers=24, d_model=1024, num_heads=16, d_ff=1024 * 4)
    gpt2s_cfg = ResourceConfig(vocab_size=50257, context_length=1024, num_layers=12, d_model=768, num_heads=12, d_ff=768 * 4)

    # increased context length
    gpt2xlxc_cfg = ResourceConfig(vocab_size=50257, context_length=16384, num_layers=48, d_model=1600, num_heads=25, d_ff=6400)
    resource_accounting(gpt2xl_cfg)
