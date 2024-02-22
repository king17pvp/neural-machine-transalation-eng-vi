import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIAL_SYMBOLS = ['<unk>', '<pad>', '<sos>', '<eos>']

class MultiHeadAttention(nn.Module):
    def __init__(self, attention_head, model_dimension):
        super(MultiHeadAttention, self).__init__()
        assert model_dimension % attention_head == 0, "dimension of model must be divisible by the attention head"

        self.attention_head = attention_head
        self.model_dimension = model_dimension
        self.d_k = self.model_dimension // self.attention_head

        # All of the below has shape
        self.W_q = nn.Linear(model_dimension, model_dimension, bias = False) # Query transformation
        self.W_k = nn.Linear(model_dimension, model_dimension, bias = False) # Key transformation
        self.W_v = nn.Linear(model_dimension, model_dimension, bias = False) # Value transformation
        self.W_o = nn.Linear(model_dimension, model_dimension) # Output transformation

    def scaled_dot_products(self, Q, K, V, mask = None):
        # Q has shape (batch_size, num_heads, n_q, d_k)
        # K has shape (batch_size, num_heads, n_k, d_k)
        # V has shape (batch_size, num_heads, n_v, d_k)
        # Where n_q, n_k, n_v are seq_len of either src / tgt sequence
        # Mask has shape (batch_size, 1, 1, src_seq_len)
        # Mask has shape (batch_size, 1, tgt_seq_len, tgt_seq_len)
        
        # After tranposing, K has shape (batch_sze, num_heads, d_k, n_k)
        attention_score = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # Attention_score has shape (batch_size, num_heads, n_q, n_k)
        
        if mask is not None:
            mask = mask.unsqueeze(1)
            attention_score = attention_score.masked_fill(mask == 0, value = -1e9)

        # Attention_probability is computed via softmax function
        # Still have shape (batch_size, num_heads, n_q, n_k)
        attention_probability = torch.softmax(attention_score, dim = -1)


        # (batch_size, num_heads, n_q, n_k) * (batch_size, num_heads, n_v, d_k)
        # In multi-head self-attention there are 2 cases:
        # If it's not cross attention, then 
        # Q = X * W_q, K = X * W_k, V = X * W_v
        # If it is then 
        # Q = X * W_q, K = encoder_output * W_k, V = encoder_output * W_k
        # => output has shape (batch_size, num_heads, n_q, d_k) 
        output = (attention_probability @ V)
        return output

    def split_heads(self, X):
        '''
        Reshape input X to have attention_head for multi-head attention
        '''

        # Tensor X has shape (batch_size, seq_len, model_dimension)

        batch_size, seq_len, model_dimension = X.shape
        output = X.view(batch_size, seq_len, self.attention_head, self.d_k)
        output = output.transpose(1, 2)
        # Return Tensor has shape (batch_size, attention_head, seq_len, dimension of each head)
        return output

    def combine_heads(self, X):
        '''
        Reshape input tensor X to have the same dimension before being fed for
        Multi head attention    
        '''
        # Tensor X has shape (batch_size, attention_head, seq_len, dim_each_head)
        # X.tranpose(1, 2) has shape (batch_size, seq_len, attention_head, dim_each_head)
        
        batch_size, _, seq_len, d_k = X.shape
        output = X.transpose(1, 2).contiguous().view(batch_size, seq_len, self.model_dimension)
        return output

    def forward(self, Q, K, V, mask = None):

        # Seems like Q, K, V all has shape (batch_size, n, d)
        # Where n is input sequence length
        # Where d is embedding dimension / model_dimension
        # To simplify this, I will consider model_dimension
        # And embedding dimension the same
        # But if embedding dimension is different then
        # We can change input dimension of Wo, Wv, Wk, Wq

        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attention_output = self.scaled_dot_products(Q, K, V, mask)

        # attention_output has shape (batch_size, num_heads, n_q, d_k)

        output = self.W_o(self.combine_heads(attention_output))
        # combine heads we have (batch_size, n_q, model_dimension)

        return output

class PositionWiseFeedForwardNetwork(nn.Module):
    def __init__(self, model_dimension, feed_forward_dimension):

        super(PositionWiseFeedForwardNetwork, self).__init__()
        self.model_dimension = model_dimension
        self.feed_forward_dimension = feed_forward_dimension

        self.fc1 = nn.Linear(model_dimension, feed_forward_dimension)
        self.fc2 = nn.Linear(feed_forward_dimension, model_dimension)
        self.relu = nn.ReLU()

    def forward(self, X):
        # X has shape (batch_size, seq_len, model_dimension)
        # Return tensor has the same thing
        return self.fc2(self.relu(self.fc1(X)))
    
class PositionalEncoding(nn.Module):
    def __init__(self, model_dimension, max_seq_len, dropout):
        super(PositionalEncoding, self).__init__()
        self.model_dimension = model_dimension
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(dropout)
        positional_encoding = torch.zeros(max_seq_len, model_dimension)
        position = torch.arange(0, max_seq_len, dtype = torch.float).unsqueeze(1)

        # position has shape (max_seq_len, 1)
        # positional_encoding has shape (max_seq_len, model_dimension)

        # log(pos) - 2i/model * log(10000)
        div_term = torch.exp(torch.arange(0, model_dimension, 2).float() * -(math.log(10000.0) / model_dimension))

        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0)
        
        self.register_buffer('pe', positional_encoding)

    def forward(self, X):
        # X has shape (batch_size, seq_len, model_dimension)
        return self.dropout(X + (self.pe[:, :X.shape[1], :]))
    
class EncoderBlock(nn.Module):
    def __init__(self, model_dimension, attention_heads, feed_forward_dimension, dropout):
        super(EncoderBlock, self).__init__()

        self.attention = MultiHeadAttention(attention_heads, model_dimension)
        self.feed_forward_network = PositionWiseFeedForwardNetwork(model_dimension, feed_forward_dimension)
        self.norm1 = nn.LayerNorm(model_dimension)
        self.norm2 = nn.LayerNorm(model_dimension)

        self.dropout = nn.Dropout(dropout)

    def forward(self, X, mask):
        # X is a tensor of shape (batch_size, seq_len, model_dimmension)
        attention_output = self.attention(X, X, X, mask)
        X = self.norm1(X + self.dropout(attention_output))
        feed_forward_output = self.feed_forward_network(X)
        X = self.norm2(X + self.dropout(feed_forward_output))

        # Return tensor is the same size
        return X


class DecoderBlock(nn.Module):
    def __init__(self, model_dimension, attention_heads, feed_forward_dimension, dropout):
        super(DecoderBlock, self).__init__()

        self.attention = MultiHeadAttention(attention_heads, model_dimension)
        self.feed_forward_network = PositionWiseFeedForwardNetwork(model_dimension, feed_forward_dimension)
        self.norm1 = nn.LayerNorm(model_dimension)
        self.norm2 = nn.LayerNorm(model_dimension)
        self.norm3 = nn.LayerNorm(model_dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, encoder_output, source_mask, target_mask):

        attention_output = self.attention(X, X, X, target_mask)
        X = self.norm1(X + self.dropout(attention_output))
        # X after add & norm layer has the same shape (batch_size, tgt_len, model_dim)    
        # In this scenario, encoder output will be played as key and value
        # This is cross-attention
        # Encoder_output has shape (batch_size, src_len, model_dim)
        attention_output = self.attention(X, encoder_output, encoder_output, source_mask)
        
        # Attention_output has shape (batch_size, tgt_len, model_dim)
        X = self.norm2(X + self.dropout(attention_output))
            
        feed_forward_output = self.feed_forward_network(X)
        X = self.norm3(X + self.dropout(feed_forward_output))

        return X
    
class Transformer(nn.Module):
    def __init__(self, model_dimension, attention_heads, feed_forward_dimension,
                 source_vocab_size, target_vocab_size, num_layers, max_seq_len, dropout):
        super(Transformer, self).__init__()

        self.model_dimension = model_dimension
        self.attention_heads = attention_heads
        self.feed_forward_dimension = feed_forward_dimension

        self.positional_encoding = PositionalEncoding(model_dimension, max_seq_len, dropout)
        self.encoder_embedding = nn.Embedding(source_vocab_size, model_dimension)
        self.decoder_embedding = nn.Embedding(target_vocab_size, model_dimension)
        self.encoders = nn.ModuleList([EncoderBlock(model_dimension, attention_heads, feed_forward_dimension, dropout) for _ in range(num_layers)])
        self.decoders = nn.ModuleList([DecoderBlock(model_dimension, attention_heads, feed_forward_dimension, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(model_dimension, target_vocab_size)
    
    
    def generate_mask(self, source_sentence, target_sentence):
        # target sentence has shape (batch_size, max_tgt_len)
        # source sentence has shape (batch_size, max_src_len)
        # since batch_size can be broadcasted so we can skip it

        batch_size = source_sentence.shape[0]
        max_target_len = target_sentence.shape[1]
        #print((target_sentence != PAD_IDX).unsqueeze(1))
        # source_sentence has shape (batch_size, 1, max_src_len)
        # = 1 if it's not PAD_IDX, 0 otherwise
        source_mask = (source_sentence != PAD_IDX).unsqueeze(1).int().to(device)
        
        # target_mask has shape (batch_size, 1, max_tgt_len)
        target_mask = (target_sentence != PAD_IDX).unsqueeze(1).int().to(device)

        # no peak mask has shape (batch_size, max_tgt_len, max_tgt_len)
        # = 0 if i >= j, 1 otherwise
        no_peak_mask = 1 - torch.triu(torch.ones((1, max_target_len, max_target_len)), diagonal=1).type(torch.int).to(device)

        # every position (i,j) such that i >= j and not a PAD_IDX
        target_mask = target_mask & no_peak_mask
        # target_mask now has shape (batch_size, max_tgt_len, max_tgt_len)

        return source_mask, target_mask

    def forward(self, source_sentence, target_sentence):

        # source sentence have shape (batch_size, src_seq_len)
        # target sentence have shape (batch_size, tgt_seq_len)
        source_mask, target_mask = self.generate_mask(source_sentence, target_sentence)
        source_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(source_sentence)))
        target_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(target_sentence)))

        # Now source_embedded have shape (batch_size, src_seq_len, model_dimension)
        # Now target_embedded have shape (batch_size, tgt_seq_len, model_dimension)
        
        encoder_output = source_embedded
        for encoder_layer in self.encoders:
            encoder_output = encoder_layer(encoder_output, source_mask)
        # encoder_output has shape (batch_size, src_seq_len, model_dimension)
        
        decoder_output = target_embedded
        for decoder_layer in self.decoders:
            decoder_output = decoder_layer(decoder_output, encoder_output, source_mask, target_mask)

        # Output will have shape (batch_size, max_seq_len, target_vocab_size)
        output = self.fc(decoder_output)
        return output