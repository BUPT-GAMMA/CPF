# pylint: disable= no-member, arguments-differ, invalid-name
# pylint: enable=W0235
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch.utils import Identity
from dgl.base import DGLError
from dgl.ops import edge_softmax
# from dgl.nn.pytorch import GATConv
from dgl.nn import DotGatConv


class PLP(nn.Module):
    def __init__(self, g, num_layers, in_dim, emb_dim, num_classes, activation,
                 feat_drop, attn_drop, residual, byte_idx_train, labels_one_hot, ptype, mlp_layers):
        super(PLP, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.plp_layers = nn.ModuleList()
        self.activation = activation
        self.node_num = byte_idx_train.shape[0]
        self.masked = byte_idx_train
        self.masked_labels_one_hot = th.mul(labels_one_hot, self.masked)
        self.alpha = nn.Parameter(th.zeros(size=(self.node_num, 1)))
        self.mlp_layers = mlp_layers
        self.plp_layer = PLPConv(
                        in_dim, emb_dim, num_classes, self.node_num, feat_drop,
                        attn_drop, residual, self.activation,
                        mlp_layers=self.mlp_layers, ptype=ptype)

    def forward(self, features, label_init):
        h = label_init
        for l in range(self.num_layers):
            h, att, alpha, el, er = self.plp_layer(self.g, features, h)
            h = th.mul(h, ~self.masked) + self.masked_labels_one_hot
        # output projection
        logits = th.mul(h, ~self.masked) + self.masked_labels_one_hot
        return logits, att, alpha, el, er


class PLPConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_class,
                 node_num,
                 feat_drop=0.,
                 attn_drop=0.,
                 residual=False,
                 activation=None,
                 mlp_layers=0,
                 allow_zero_in_degree=False,
                 ptype='ind'):
        super(PLPConv, self).__init__()
        self._in_src_feats = in_feats
        self._out_feats = out_feats
        self.ptype = ptype
        self.lr_alpha = nn.Parameter(th.zeros(size=(node_num, 1)))
        self._allow_zero_in_degree = allow_zero_in_degree
        if self.ptype == 'ind':
            self.attn_l = nn.Parameter(th.FloatTensor(size=(1, in_feats)))
        elif self.ptype == 'tra':
            self.fc_emb = nn.Parameter(th.FloatTensor(size=(node_num, 1)))
        else:
            raise ValueError(r'No such ptype!')
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.mlp_layers = mlp_layers
        if self.mlp_layers > 0:
            self.mlp = MLP2(self.mlp_layers, self._in_src_feats, out_feats, num_class, feat_drop)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain('relu')
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
        if self.ptype == 'ind':
            nn.init.xavier_normal_(self.attn_l, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_emb, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, soft_label):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')
            if self.ptype == 'ind':
                feat_src = h_dst = self.feat_drop(feat)
                el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
                er = th.zeros(graph.num_nodes(), device=graph.device)
            elif self.ptype == 'tra':
                feat_src = self.feat_drop(self.fc_emb)
                feat_dst = h_dst = th.zeros(graph.num_nodes(), device=graph.device)
                el = feat_src
                er = feat_dst
            cog_label = soft_label
            graph.srcdata.update({'ft': cog_label, 'el': el})
            graph.dstdata.update({'er': er})
            # # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            # graph.edata['e'] = th.ones(graph.num_edges(), device=graph.device)  # non-parameterized PLP
            e = graph.edata.pop('e')
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            att = graph.edata['a'].squeeze()
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            if self.mlp_layers > 0:
                rst = th.sigmoid(self.lr_alpha) * graph.dstdata['ft'] + \
                      th.sigmoid(-self.lr_alpha) * self.mlp(feat)
            else:
                rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst)
                rst = rst + resval
            # activation
            if self.activation:
                rst = self.activation(rst)
            return rst, att, th.sigmoid(self.lr_alpha).squeeze(), el.squeeze(), er.squeeze()
            # return rst, att, self.lr_alpha.squeeze()


class DotGatConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_class,
                 node_num,
                 allow_zero_in_degree=False):
        super(DotGatConv, self).__init__()
        self._in_src_feats = self._in_dst_feats = in_feats
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.fc = nn.Linear(self._in_src_feats, self._out_feats, bias=False)
        self.fc2 = nn.Linear(self._in_src_feats, num_class, bias=False)
        self.lr_alpha = nn.Parameter(th.zeros(size=(node_num, 1)))

    def forward(self, graph, feat, soft_label):
        graph = graph.local_var()

        if not self._allow_zero_in_degree:
            if (graph.in_degrees() == 0).any():
                raise DGLError('There are 0-in-degree nodes in the graph, '
                               'output for those nodes will be invalid. '
                               'This is harmful for some applications, '
                               'causing silent performance regression. '
                               'Adding self-loop on the input graph by '
                               'calling `g = dgl.add_self_loop(g)` will resolve '
                               'the issue. Setting ``allow_zero_in_degree`` '
                               'to be `True` when constructing this module will '
                               'suppress the check and let the code run.')
        h_src = feat
        feat_src = feat_dst = self.fc(h_src)
        if graph.is_block:
            feat_dst = feat_src[:graph.number_of_dst_nodes()]

        # Assign features to nodes
        graph.srcdata.update({'ft': feat_src})
        graph.dstdata.update({'ft': feat_dst})
        # Step 1. dot product
        graph.apply_edges(fn.u_dot_v('ft', 'ft', 'a'))
        # graph.edata['a'] = th.ones(graph.num_edges(), device=graph.device)
        # Step 2. edge softmax to compute attention scores
        graph.edata['sa'] = edge_softmax(graph, graph.edata['a'])
        att = graph.edata['sa'].squeeze()
        cog_label = soft_label
        # cog_label = self.fc2(feat)
        # cog_label = th.sigmoid(self.lr_alpha) * soft_label + th.sigmoid(-self.lr_alpha) * self.fc2(feat)
        graph.srcdata.update({'ft': cog_label})
        graph.dstdata.update({'ft': cog_label})
        # Step 3. Broadcast softmax value to each edge, and aggregate dst node
        graph.update_all(fn.u_mul_e('ft', 'sa', 'attn'), fn.sum('attn', 'agg_u'))
        # output results to the destination nodes
        rst = graph.dstdata['agg_u']

        return rst, att, th.sigmoid(self.lr_alpha).squeeze()


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(MLP, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.lr1 = nn.Linear(input_dim, hidden_dim)
        self.lr2 = nn.Linear(hidden_dim, output_dim)
        # self.lr3 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        tmp1 = F.relu(self.lr1(self.dropout(x)))
        return self.lr2(self.dropout(tmp1))
        # return self.lr3(self.dropout(x))


class MLP2(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout):
        super(MLP2, self).__init__()
        self.linear_or_not = True       # default is linear model
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model ()
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = th.nn.ModuleList()
            # self.batch_norms = torch.nn.ModuleList()
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            # for layer in range(num_layers - 1):
            #     self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:  # If linear model
            # return F.log_softmax(self.linear(x), dim=1)
            # return self.linear(x)
            return self.linear(self.dropout(x))
        else:                   # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.linears[layer](self.dropout(h)))
                # h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            # return F.log_softmax(self.linears[self.num_layers - 1](self.dropout(h)), dim=1)
            return self.linears[self.num_layers - 1](self.dropout(h))
