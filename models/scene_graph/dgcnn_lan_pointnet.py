#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Joy Hsu <joycj@stanford.edu>
#
# Distributed under terms of the MIT license.

import os
import collections
from itertools import product

import torch
import torch.nn as nn
import jactorch
import jactorch.nn as jacnn


from models.scene_graph.point_net_pp_model import PointNetPP, MLP
from models.scene_graph.dgcnn import DGCNN
from models.scene_graph.lstm import LSTMEncoder
from models.scene_graph.word_embeddings import load_glove_pretrained_embedding, make_pretrained_embedding, Vocabulary

DEBUG = bool(int(os.getenv('DEBUG_SCENE_GRAPH', 0)))

__all__ = ['DGCNNPointNet', 'DGCNNFeature']


class DGCNNLanFeature(collections.namedtuple(
    '_SceneGraphFeature', ('scene_feature', 'object_feature', 'relation_feature', 'multi_relation_feature')
)):
    pass


class DGCNNLanPointNet(nn.Module):
    def __init__(self, pn_output_dim, output_dim, vocab, mode='ol'):
        super().__init__()
        
        self.pn_output_dim = pn_output_dim
        self.output_dim = output_dim
                
        self.single_object_encoder = self.single_object_encoder_init(pn_output_dim)
        self.single_object_encoder_for_relation = self.single_object_sparse_encoder_init(pn_output_dim)
        self.object_mlp = MLP(pn_output_dim, [128, 256, 607], dropout_rate=0.35)

        self.graph_encoder = DGCNN(initial_dim=pn_output_dim*2,
                              out_dim=pn_output_dim,
                              k_neighbors=3,
                              intermediate_feat_dim=[128, 128, 128, 128],
                              subtract_from_self=True)

        self.lang_encoder = token_encoder(vocab=vocab,
                                 word_embedding_dim=64,
                                 lstm_n_hidden=128,
                                 word_dropout=0.1,
                                 random_seed=42)
        
        self.relation_dense_layer = nn.Sequential(nn.ReLU(True), nn.Linear(pn_output_dim*2, pn_output_dim))
        self.relation_dense_layer_2 = nn.Sequential(nn.ReLU(True), nn.Linear(pn_output_dim, pn_output_dim))
        
        self.multi_relation_dense_layer = nn.Sequential(nn.ReLU(True), nn.Linear(pn_output_dim, pn_output_dim))
        self.multi_relation_dense_layer_2 = nn.Sequential(nn.ReLU(True), nn.Linear(pn_output_dim, pn_output_dim))


    def single_object_encoder_init(self, out_dim: int) -> PointNetPP:
        """
        The default PointNet++ encoder for a 3D object.
        @param: out_dims: The dimension of each object feature
        """
        return PointNetPP(sa_n_points=[32, 16, None],
                          sa_n_samples=[32, 32, None],
                          sa_radii=[0.2, 0.4, None],
                          sa_mlps=[[3, 64, 64, 128],
                                   [128, 128, 128, 256],
                                   [256, 256, 512, out_dim]])
    
    def single_object_sparse_encoder_init(self, out_dim: int) -> PointNetPP:
        """
        The default PointNet++ encoder for a 3D object.
        @param: out_dims: The dimension of each object feature
        """
        return PointNetPP(sa_n_points=[8, None],
                          sa_n_samples=[8, None],
                          sa_radii=[0.4, None],
                          sa_mlps=[[3, 32, 32, 64],
                                   [64, 64, 64, out_dim]])
    
    def union(self, cloud1, cloud2):
        return torch.concat([cloud1, cloud2], dim=1)
    
    def mask_for_location_only(self, target):
        masked_channels = torch.zeros((target.size(0), target.size(1), 3))
        masked_full = torch.concat([target[:, :, :3], masked_channels.cuda()], dim=-1)    
        return masked_full

    def forward(
        self,
        input: torch.Tensor,
        objects: torch.Tensor,
        objects_length: torch.Tensor,
        lang: torch.Tensor
    ):
        scene = input
        objects_length = objects_length.detach().cpu()
        max_nr_objects = objects_length.max().item()       
        
        '''For relation embedding'''
        pn_object_feat = []
        for i in range(objects.size(1)):
            masked_object = self.mask_for_location_only(objects[:, i, :, :])
            curr_single_obj_feat = self.single_object_encoder_for_relation(masked_object)
            pn_object_feat.append(curr_single_obj_feat)
        pn_object_feat = torch.stack(pn_object_feat) 
        pn_object_feat = pn_object_feat.permute(1, 0, 2)
        lang_feat = self.lang_encoder(lang)
        lang_features_expanded = torch.unsqueeze(lang_feat, -1).expand(-1, -1, pn_object_feat.shape[1]).transpose(2, 1)
        graph_in_features = torch.cat([pn_object_feat, lang_features_expanded], dim=-1)
        graph_out_features = self.graph_encoder(graph_in_features)
        pn_object_feat = graph_out_features.permute(1, 0, 2)
        
        relation_feat = [[0 for i in range(objects.size(1))] for j in range(objects.size(1))]
        multi_relation_feat = [[0 for i in range(objects.size(1))] for j in range(objects.size(1))]
        for i, j in product(range(objects.size(1)), range(objects.size(1))):
            union_object_feat = self.union(pn_object_feat[i], pn_object_feat[j])
            union_object_feat = self.relation_dense_layer(union_object_feat)
            union_object_feat = self.relation_dense_layer_2(union_object_feat)           
            relation_feat[i][j] = union_object_feat
            
            multi_union_object_feat = self.multi_relation_dense_layer(union_object_feat)
            multi_union_object_feat = self.multi_relation_dense_layer_2(multi_union_object_feat)
            multi_relation_feat[i][j] = multi_union_object_feat
                
        relation_for_o = []
        for o in relation_feat:
            relation_for_o.append(torch.stack(o))
        relation_feat = torch.stack(relation_for_o) 
            
        multi_relation_for_o = []
        for o in multi_relation_feat:
            multi_relation_for_o.append(torch.stack(o))
        multi_relation_feat = torch.stack(multi_relation_for_o)
        
        '''For object embedding'''
        object_feat = []
        for i in range(objects.size(1)):      
            curr_single_obj_feat = self.single_object_encoder(objects[:, i, :, :])
            curr_single_obj_feat = self.object_mlp(curr_single_obj_feat)
            object_feat.append(curr_single_obj_feat)                
        object_feat = torch.stack(object_feat)
       
        outputs = list()
        for i in range(objects.size(0)):
            this_object_feat = object_feat[:, i, :]
            this_relation_feat = relation_feat[:, :, i, :]
            this_multi_relation_feat = multi_relation_feat[:, :, i, :]
            outputs.append(DGCNNLanFeature(None, this_object_feat, this_relation_feat, this_multi_relation_feat))
                            
        return outputs

    def _norm(self, x):
        return x / x.norm(2, dim=-1, keepdim=True)

def token_encoder(vocab: Vocabulary,
                  word_embedding_dim: int,
                  lstm_n_hidden: int,
                  word_dropout: float,
                  init_c=None, init_h=None, random_seed=None,
                  feature_type='max',
                  glove_emb_file: str = '') -> LSTMEncoder:
    """
    Language Token Encoder.

    @param vocab: The vocabulary created from the dataset (nr3d or sr3d) language tokens
    @param word_embedding_dim: The dimension of each word token embedding
    @param glove_emb_file: If provided, the glove pretrained embeddings for language word tokens
    @param lstm_n_hidden: The dimension of LSTM hidden state
    @param word_dropout:
    @param init_c:
    @param init_h:
    @param random_seed:
    @param feature_type:
    """
    if len(glove_emb_file) > 0:
        print('Using glove pre-trained embeddings.')
        glove_embedding = load_glove_pretrained_embedding(glove_emb_file, verbose=True)
        word_embedding = make_pretrained_embedding(vocab, glove_embedding, random_seed=random_seed)

        # word-projection here is a bit deeper, since the glove-embedding is frozen.
        word_projection = nn.Sequential(nn.Linear(word_embedding_dim, word_embedding_dim),
                                        nn.ReLU(),
                                        nn.Dropout(word_dropout),
                                        nn.Linear(word_embedding_dim, word_embedding_dim),
                                        nn.ReLU())
    else:
        word_embedding = nn.Embedding(len(vocab), word_embedding_dim, padding_idx=vocab.pad)
        word_projection = nn.Sequential(nn.Dropout(word_dropout),
                                        nn.Linear(word_embedding_dim, word_embedding_dim),
                                        nn.ReLU())

    assert vocab.pad == 0 and vocab.eos == 2

    model = LSTMEncoder(n_input=word_embedding_dim, n_hidden=lstm_n_hidden, word_embedding=word_embedding,
                        init_c=init_c, init_h=init_h, word_transformation=word_projection, eos_symbol=vocab.eos,
                        feature_type=feature_type)
    return model
