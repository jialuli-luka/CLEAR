
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from param import args
from transformers import BertModel, BertConfig
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderLSTM(nn.Module):
    ''' Encodes navigation instructions, returning hidden state context (for
        attention methods) and a decoder initial state. '''

    def __init__(self, vocab_size, embedding_size, hidden_size, padding_idx, 
                            dropout_ratio, bidirectional=False, num_layers=1):
        super(EncoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(p=dropout_ratio)
        if bidirectional:
            print("Using Bidir in EncoderLSTM")
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx)
        input_size = embedding_size
        self.lstm = nn.LSTM(input_size, hidden_size, self.num_layers,
                            batch_first=True, dropout=dropout_ratio, 
                            bidirectional=bidirectional)
        self.encoder2decoder = nn.Linear(hidden_size * self.num_directions,
            hidden_size * self.num_directions
        )

    def init_state(self, inputs):
        ''' Initialize to zero cell states and hidden states.'''
        batch_size = inputs.size(0)
        h0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        c0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)

        return h0.to(device), c0.to(device)

    def forward(self, inputs, lengths):
        ''' Expects input vocab indices as (batch, seq_len). Also requires a 
            list of lengths for dynamic batching. '''
        embeds = self.embedding(inputs)  # (batch, seq_len, embedding_size)
        embeds = self.drop(embeds)
        h0, c0 = self.init_state(inputs)
        packed_embeds = pack_padded_sequence(embeds, lengths, batch_first=True)
        enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0, c0))

        if self.num_directions == 2:    # The size of enc_h_t is (num_layers * num_directions, batch, hidden_size)
            h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
            c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
        else:
            h_t = enc_h_t[-1]
            c_t = enc_c_t[-1] # (batch, hidden_size)

        ctx, _ = pad_packed_sequence(enc_h, batch_first=True)

        if args.sub_out == "max":
            ctx_max, _ = ctx.max(1)
            decoder_init = nn.Tanh()(self.encoder2decoder(ctx_max))
        elif args.sub_out == "tanh":
            decoder_init = nn.Tanh()(self.encoder2decoder(h_t))
        else:
            assert False

        ctx = self.drop(ctx)
        if args.zero_init:
            return ctx, torch.zeros_like(decoder_init), torch.zeros_like(c_t)
        else:
            return ctx, decoder_init, c_t  # (batch, seq_len, hidden_size*num_directions)
                                 # (batch, hidden_size)


class EncoderMBert(nn.Module):
    def __init__(self, hidden_size):
        super(EncoderMBert, self).__init__()
        self.encoder2decoder = nn.Linear(hidden_size, hidden_size)

    def forward(self, inputs):
        # print("\tIn Model: input size", inputs.size())
        h_t = inputs[:, 0, :]
        decoder_init = nn.Tanh()(self.encoder2decoder(h_t))
        c_t = decoder_init
        return inputs, decoder_init, c_t


class SoftDotAttention(nn.Module):
    '''Soft Dot Attention. 

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    '''

    def __init__(self, query_dim, ctx_dim):
        '''Initialize layer.'''
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(query_dim, ctx_dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(query_dim + ctx_dim, query_dim, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, h, context, mask=None,
                output_tilde=True, output_prob=True):
        '''Propagate h through the network.

        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        logit = attn

        if mask is not None:
            # -Inf masking prior to the softmax
            attn.masked_fill_(mask, -float('inf'))
        attn = self.sm(attn)    # There will be a bug here, but it's actually a problem in torch source code.
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        if not output_prob:
            attn = logit
        if output_tilde:
            h_tilde = torch.cat((weighted_context, h), 1)
            h_tilde = self.tanh(self.linear_out(h_tilde))
            return h_tilde, attn
        else:
            return weighted_context, attn


class AttnDecoderLSTM(nn.Module):
    ''' An unrolled LSTM with attention over instructions for decoding navigation actions. '''

    def __init__(self, embedding_size, hidden_size,
                       dropout_ratio, feature_size=2048+4):
        super(AttnDecoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.embedding = nn.Sequential(
            nn.Linear(args.angle_feat_size, self.embedding_size),
            nn.Tanh()
        )
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop_env = nn.Dropout(p=args.featdropout)
        self.lstm = nn.LSTMCell(embedding_size+feature_size, hidden_size)
        self.feat_att_layer = SoftDotAttention(hidden_size, feature_size)
        self.attention_layer = SoftDotAttention(hidden_size, hidden_size)
        self.candidate_att_layer = SoftDotAttention(hidden_size, feature_size)

    def forward(self, action, feature_in, cand_feat_in,
                h_0, prev_h1, c_0,
                ctx, ctx_mask=None,
                already_dropfeat=False,
                pair_ctx=None, pair_mask=None):
        '''
        Takes a single step in the decoder LSTM (allowing sampling).
        action: batch x angle_feat_size
        feature: batch x 36 x (feature_size + angle_feat_size)
        cand_feat: batch x cand x (feature_size + angle_feat_size)
        h_0: batch x hidden_size
        prev_h1: batch x hidden_size
        c_0: batch x hidden_size
        ctx: batch x seq_len x dim
        ctx_mask: batch x seq_len - indices to be masked
        already_dropfeat: used in EnvDrop
        '''
        action_embeds = self.embedding(action)

        # Adding Dropout
        action_embeds = self.drop(action_embeds)

        if not already_dropfeat:
            # Dropout the raw feature as a common regularization
            feature = feature_in.clone()
            feature[..., :-args.angle_feat_size] = self.drop_env(feature_in[..., :-args.angle_feat_size])   # Do not drop the last args.angle_feat_size (position feat)

        prev_h1_drop = self.drop(prev_h1)
        attn_feat, visual_alpha = self.feat_att_layer(prev_h1_drop, feature, output_tilde=False)

        concat_input = torch.cat((action_embeds, attn_feat), 1) # (batch, embedding_size+feature_size)
        h_1, c_1 = self.lstm(concat_input, (prev_h1, c_0))

        h_1_drop = self.drop(h_1)
        h_tilde, alpha = self.attention_layer(h_1_drop, ctx, ctx_mask)

        if pair_ctx is not None:
            weighted_ctx, _ = self.attention_layer(h_1_drop, ctx, ctx_mask, output_tilde=False)
            weighted_pair, _ = self.attention_layer(h_1_drop, pair_ctx, pair_mask, output_tilde=False)
        else:
            weighted_ctx = h_tilde
            weighted_pair = h_tilde

        # Adding Dropout
        h_tilde_drop = self.drop(h_tilde)

        if not already_dropfeat:
            cand_feat = cand_feat_in.clone()
            cand_feat[..., :-args.angle_feat_size] = self.drop_env(cand_feat_in[..., :-args.angle_feat_size])

        _, logit = self.candidate_att_layer(h_tilde_drop, cand_feat, output_prob=False)

        if args.save_attention:
            return h_1, c_1, logit, h_tilde, weighted_ctx, weighted_pair, alpha, visual_alpha
        else:
            return h_1, c_1, logit, h_tilde, weighted_ctx, weighted_pair


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.state2value = nn.Sequential(
            nn.Linear(args.bert_dim, args.bert_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.bert_dim, 1),
        )

    def forward(self, state):
        return self.state2value(state).squeeze()


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.register_buffer("temperature", torch.tensor(temperature))
        self.batch_size = args.batchSize
        self.sm = nn.Softmax()
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignoreid, size_average=False)

    def forward(self, emb_i, emb_j, label=None):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)      # (Batch*2) X BERT_dim
        if args.sim == 'cosine':
            similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        elif args.sim == 'l2':
            differences = representations.unsqueeze(1) - representations.unsqueeze(0)
            similarity_matrix = torch.sum(differences * differences, -1)
        elif args.sim == 'ce':
            similarity_matrix_i = F.cosine_similarity(z_i.unsqueeze(1), z_j.unsqueeze(0), dim=2)
            similarity_matrix_j = F.cosine_similarity(z_j.unsqueeze(1), z_i.unsqueeze(0), dim=2)
        else:
            print("Wrong similarity method")
            exit()

        def l_ij(i, j):
            sim_i_j = similarity_matrix[i, j]

            numerator = torch.exp(sim_i_j / self.temperature)
            one_for_not_i = torch.ones((2 * self.batch_size,)).to(device).scatter_(0, torch.tensor([i]).to(device), 0.0)

            denominator = torch.sum(
                    one_for_not_i * torch.exp(similarity_matrix[i, :] / self.temperature)
                )

            loss_ij = -torch.log(numerator / denominator)

            return loss_ij.squeeze(0)

        N = emb_i.size(0)
        loss = 0.0
        if args.sim == 'ce':
            loss += self.criterion(similarity_matrix_i / self.temperature, label)
            loss += self.criterion(similarity_matrix_j / self.temperature, label)
        else:
            for k in range(0, N):
                loss += l_ij(k, k + N)
            loss = loss * 2
        return 1.0 / (2 * N) * loss


class VisualEncoder(nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self.feature_size = feature_size
        self.ff_layer = nn.Sequential(
            nn.Dropout(p=args.featdropout),
            nn.Linear(self.feature_size, self.feature_size),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(self.feature_size, self.feature_size),
        )
        self.norm = nn.LayerNorm(self.feature_size)

    def forward(self, feature, objects=None):
        if args.load_visual is None:
            if args.visual_before:
                input = torch.mean(feature, dim=1)
            else:
                batch, view, dim = feature.shape
                input = feature.view(-1, dim)
        else:
            input = feature
        embed_feature = self.ff_layer(input)
        output = self.norm(embed_feature + input)
        if args.load_visual is None and not args.visual_before:
            output = output.view(feature.shape)
            if objects is not None:
                output.masked_fill_(objects, 0.0)
            output = torch.mean(output, dim=1)

        return output


class VisualAttention(nn.Module):
    def __init__(self, feature_size, hidden_size):
        super().__init__()
        self.feat_att_layer = SoftDotAttention(hidden_size, feature_size)

    def forward(self, text, visual):
        attn_feat, _ = self.feat_att_layer(text, visual, output_tilde=False)

        return attn_feat


class VisualCombine(nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self.feature_size = feature_size
        self.private_layer_d = nn.Linear(self.feature_size, self.feature_size, bias=False)
        self.shared_layer_d = nn.Linear(self.feature_size, self.feature_size, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.private_layer_u = nn.Linear(self.feature_size, self.feature_size, bias=False)
        self.shared_layer_u = nn.Linear(self.feature_size, self.feature_size, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, shared, private):
        d = self.sigmoid(self.private_layer_d(private) + self.shared_layer_d(shared))
        dot_product = torch.mul(private, d)
        out = self.tanh(self.private_layer_u(dot_product) + self.shared_layer_u(shared))

        return out




