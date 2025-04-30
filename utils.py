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

if __name__ == '__main__':
    print(get_sinusoidal_position_embeddings(8, 4))
