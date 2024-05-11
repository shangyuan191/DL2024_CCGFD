
import torch
import torch.nn as nn
import torch.nn.functional as F


class _attention(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, output_attn=False, is_causal=False, feed_forward=nn.Linear, use_weight=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads

        self.output_attn = output_attn
        self.is_causal = is_causal
        self.feed_forward = feed_forward
        self.use_weight = use_weight

        self.Wk = feed_forward(in_channels, out_channels * num_heads)
        self.Wq = feed_forward(in_channels, out_channels * num_heads)
        if use_weight:
            self.Wv = feed_forward(in_channels, out_channels * num_heads)

    def _input_feed_forward(self, query_input, source_input):
        query = self.Wq(query_input).reshape(-1, self.num_heads, self.out_channels)
        key = self.Wk(source_input).reshape(-1, self.num_heads, self.out_channels)
        if self.use_weight:
            value = self.Wv(source_input).reshape(-1, self.num_heads, self.out_channels)
        else:
            value = source_input.reshape(-1, 1, self.out_channels)
        return query, key, value

    def forward(self, query_input, source_input):

        pass

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        if self.use_weight:
            self.Wv.reset_parameters()

class simple_attention(_attention):
    def max_mean_attention(self):
        pass
    def mean_attention(self):
        pass
    def simple_attention(self, qs, ks, vs):
        qs = qs / torch.norm(qs, p=2)
        ks = ks / torch.norm(ks, p=2)
        N = qs.shape[0]

        kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
        attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs)  # [N, H, D]
        all_ones = torch.ones([vs.shape[0]]).to(vs.device)
        vs_sum = torch.einsum("l,lhd->hd", all_ones, vs)  # [H, D]
        attention_num += vs_sum.unsqueeze(0).repeat(vs.shape[0], 1, 1)  # [N, H, D]

        # denominator
        all_ones = torch.ones([ks.shape[0]]).to(ks.device)
        ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
        attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)  # [N, H]

        # attentive aggregated results
        attention_normalizer = torch.unsqueeze(attention_normalizer, len(attention_normalizer.shape))  # [N, H, 1]
        attention_normalizer += torch.ones_like(attention_normalizer) * N
        attn_output = attention_num / attention_normalizer  # [N, H, D]

        # compute attention for visualization if needed
        # this not work yet
        if self.output_attn:
            attention = torch.einsum("nhm,lhm->nlh", qs, ks).mean(dim=-1)
            normalizer = attention_normalizer.squeeze(dim=-1).mean(dim=-1, keepdims=True)
            attention = attention / normalizer
            #attention = torch.einsum("nhm,lhm->nlh", qs, ks).mean(dim=2) / attention_normalizer.mean(dim=1).reshape(-1)  # [N, L, H]
            return attn_output, attention

        return attn_output

    def forward(self, query_input, source_input):
        qs, ks, vs = self._input_feed_forward(query_input, source_input)
        if self.output_attn:
            out, att = self.simple_attention(qs, ks, vs)
            return out, att
        out = self.simple_attention(qs, ks, vs)
        return out




class flash_attention(_attention):
    def forward(self, query_input, source_input):
        qs, ks, vs = self._input_feed_forward(query_input, source_input)
        #out = self.simple_attention(qs, ks, vs)
        out = out.mean(dim=1)
        return out



if __name__ == '__main__':
    batch = 8
    i_dim = 512
    o_dim = 128
    head = 4
    x = torch.randn(batch, i_dim).to("cuda")

    simple_model = simple_attention(i_dim, o_dim, head, True).to("cuda")

    simple_y = simple_model(x, x)
    print(simple_y[1].shape)

#def flash_attention(qs, ks, vs, output_attn=False):

