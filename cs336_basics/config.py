from dataclasses import dataclass

@dataclass
class Config:
    
    def __init__(self, path: str | None=None):
        """reads config from file or initializes with the variables shown below"""
        
        if path:
            raise NotImplementedError()
        else:
            optimizer = 

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--optimizer", type=torch.optim.Optimizer, help="Optimizer to use"
    )


class Config:
    def __init__(self, vocab_size, context_length, num_layers, d_model, num_heads, d_ff, glu=False):
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.context_length = context_length
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.glu = glu