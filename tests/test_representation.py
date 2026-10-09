from config import load_config

import torch
from losses.representation import contrastive_loss, loss_representation
import numpy as np
from math import log, exp

CONFIG = load_config()

def util_sign(result):
    assert result['L_rep_u'].item() >= 0.0
    assert result['L_rep_s'].item() >= 0.0

def test_contract():
    features = torch.randn(2*CONFIG.batch_size_train, CONFIG.out_dim)
    labels = torch.randint(0, 100, (CONFIG.batch_size_train, ))
    is_lab = torch.rand(CONFIG.batch_size_train) > 0.5

    result = loss_representation(features, labels, is_lab, CONFIG.tau_u, CONFIG.tau_c)

    assert type(result) == dict
    assert len(result) == 2
    assert 'L_rep_u' in result and result['L_rep_u'].dtype == torch.float32
    assert 'L_rep_s' in result and result['L_rep_s'].dtype == torch.float32
    util_sign(result)

def test_equal_features():
    B, B_l = 8, 3
    features = torch.ones(2*B, CONFIG.out_dim)
    labels = torch.randint(0, 100, (B, ))
    is_lab = torch.arange(B) < B_l

    result = loss_representation(features, labels, is_lab, CONFIG.tau_u, CONFIG.tau_c)

    assert np.isclose(result['L_rep_u'].item(), log(2*B-1))
    assert np.isclose(result['L_rep_s'].item(), log(2*B_l-1))
    util_sign(result)

def test_no_lab():
    features = torch.randn(2*CONFIG.batch_size_train, CONFIG.out_dim)
    labels = torch.randint(0, 100, (CONFIG.batch_size_train, ))
    is_lab = torch.zeros(CONFIG.batch_size_train).bool()

    result = loss_representation(features, labels, is_lab, CONFIG.tau_u, CONFIG.tau_c)

    assert 'L_rep_s' in result and result['L_rep_s'].dtype == torch.float32
    util_sign(result)

def test_scale():
    features = torch.randn(2*CONFIG.batch_size_train, CONFIG.out_dim)
    features_bis = 4.2 * features
    labels = torch.randint(0, 100, (CONFIG.batch_size_train, ))
    is_lab = torch.rand(CONFIG.batch_size_train) > 0.5

    result = loss_representation(features, labels, is_lab, CONFIG.tau_u, CONFIG.tau_c)
    result_bis = loss_representation(features_bis, labels, is_lab, CONFIG.tau_u, CONFIG.tau_c)

    assert np.isclose(result['L_rep_u'].item(), result_bis['L_rep_u'].item())
    assert np.isclose(result['L_rep_s'].item(), result_bis['L_rep_s'].item())
    util_sign(result)
    util_sign(result_bis)

def test_ortho():
    B, D = 8, 8
    E = torch.eye(B, dtype=torch.float32)
    features = torch.cat((E, E))
    labels = torch.arange(B)
    is_lab = torch.ones(B).bool()

    for tau in [1.0, 0.07]:
        result = loss_representation(features, labels, is_lab, tau, tau)

        assert np.isclose(result['L_rep_u'].item(), log(1+(2*B-2)*exp(-1/tau)), atol=1e-7)
        assert np.isclose(result['L_rep_u'].item(), result['L_rep_s'].item())
        util_sign(result)

def test_hardcoded_supcon():
    features = torch.zeros(6, 10)
    labels = torch.Tensor([7, 2, 7]).to(torch.int64)
    is_lab = torch.Tensor([1, 1, 1]).bool()
    
    for i in range(len(labels)):
        features[i, labels[i].item()] = 1
        features[i+3, labels[i].item()] = 1
    
    result = loss_representation(features, labels, is_lab, CONFIG.tau_u, 1.0)
    assert np.isclose(result['L_rep_s'].item(), 1.18025)
    util_sign(result)
    
    result = loss_representation(features, labels, is_lab, CONFIG.tau_u, 0.07)
    assert np.isclose(result['L_rep_s'].item(), 0.73241)
    util_sign(result)
    
def test_add_unlabelled():
    B, B_l = 64, 48
    B_ul = B-B_l
    
    features_labelled_view1 = torch.randn(B_l, CONFIG.out_dim)
    features_labelled_view2 = torch.randn(B_l, CONFIG.out_dim)
    labels_labelled = torch.randint(0, 100, (B_l, )) 
    is_lab_labelled = torch.ones(B_l).bool()

    features_unlabelled_view1 = torch.randn(B_ul, CONFIG.out_dim)
    features_unlabelled_view2 = torch.randn(B_ul, CONFIG.out_dim)
    labels_unlabelled = torch.randint(0, 100, (B_ul, ))
    is_lab_unlabelled = torch.zeros(B_ul).bool()

    features_first = torch.cat(
        (features_labelled_view1, features_labelled_view2)
    )
    labels_first = labels_labelled
    is_lab_first = is_lab_labelled

    features_second = torch.cat(
        (features_labelled_view1, features_unlabelled_view1, features_labelled_view2, features_unlabelled_view2)
    )
    labels_second = torch.cat((labels_labelled, labels_unlabelled))
    is_lab_second = torch.cat((is_lab_labelled, is_lab_unlabelled))

    result_first = loss_representation(features_first, labels_first, is_lab_first, CONFIG.tau_u, CONFIG.tau_c)
    result_second = loss_representation(features_second, labels_second, is_lab_second, CONFIG.tau_u, CONFIG.tau_c)

    assert np.isclose(result_first['L_rep_s'].item(), result_second['L_rep_s'].item())
    util_sign(result_first)
    util_sign(result_second)

def test_backward():
    B = 8
    features = torch.cat((torch.eye(B), torch.eye(B))).requires_grad_()
    labels = torch.arange(B) % 3
    is_lab = torch.arange(B) % 2 == 0
    
    result = loss_representation(features, labels, is_lab, 0.07, 0.07)
    
    (result['L_rep_u'] + result['L_rep_s']).backward()
    assert torch.isfinite(features.grad).all()
    assert features.grad.abs().sum() > 0
