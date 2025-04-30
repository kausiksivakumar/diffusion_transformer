import torch 

def get_sinusoidal_position_embeddings(token_dim: int, embedding_dim:int) -> torch.Tensor:
    T = token_dim
    d = embedding_dim
    
    position = torch.arange(start=0, end=T).unsqueeze(1)
    denominator = torch.pow(10000,2*torch.arange(start=0, end=d//2)/d)

    pos_embeddings = torch.zeros((T,d))

    # All even nums are sin
    pos_embeddings[:,0::2]  =    torch.sin(position/denominator)
    pos_embeddings[:,1::2]  =   torch.cos(position/denominator)

    return pos_embeddings

def get_timestep_embedding(t:torch.Tensor, embedding_dim:int=128) -> torch.Tensor:
    assert len(t.shape) == 1, f"Passed value should be a flattened tensor t, got {t.shape}"
    t = t.unsqueeze(1)
    denominator = torch.pow(10000, 2*torch.arange(0, embedding_dim//2)/embedding_dim)
    timestep_embedding = torch.zeros(t.shape[0], embedding_dim)
    timestep_embedding[:,0::2] = torch.sin(t/denominator)
    timestep_embedding[:,1::2] = torch.cos(t/denominator)
    return timestep_embedding

if __name__ == '__main__':
    # print(get_sinusoidal_position_embeddings(8, 4))
    print(get_timestep_embedding(torch.tensor([0,1]), 128))
