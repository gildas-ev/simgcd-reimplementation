from config import CONFIG
from losses.classification import teacher_temp, loss_classification

import math
import numpy as np
import torch
import torch.nn.functional as F

def test_temp():
    assert teacher_temp(0, CONFIG) == CONFIG.warmup_tau_t
    assert teacher_temp(CONFIG.warmup_epochs-1, CONFIG) == CONFIG.tau_t
    assert teacher_temp(CONFIG.warmup_epochs, CONFIG) == CONFIG.tau_t
    assert teacher_temp(CONFIG.warmup_epochs+50, CONFIG) == CONFIG.tau_t

def test_contract():
    logits = torch.randn(2*CONFIG.batch_size_train, 100)
    labels = torch.randint(0, 100, (CONFIG.batch_size_train,))
    is_lab = torch.zeros(CONFIG.batch_size_train, dtype=torch.bool)

    result = loss_classification(logits, labels, is_lab,
                                 CONFIG.tau_s, CONFIG.tau_t,
                                 CONFIG.epsilon)

    assert type(result) == dict
    assert len(result) == 3
    assert 'L_cls_u' in result and result['L_cls_u'].dtype == torch.float32
    assert 'L_cls_s' in result and result['L_cls_s'].dtype == torch.float32
    assert result['L_cls_s'].item() == 0.0
    assert 'logs' in result and type(result['logs']) == dict

def test_entropy():
    logits = torch.ones(2*CONFIG.batch_size_train, 100)
    labels = torch.zeros(CONFIG.batch_size_train, dtype=torch.int64)
    is_lab = torch.rand(CONFIG.batch_size_train) > 0.5
    
    cls = loss_classification(logits, labels, is_lab,
                              CONFIG.tau_s, CONFIG.tau_t,
                              CONFIG.epsilon)

    assert np.isclose(cls['logs']['cls/entropy'].item(), math.log(100))

def test_inv():
    view_1 = torch.randn(CONFIG.batch_size_train, 100)
    view_2 = torch.randn(CONFIG.batch_size_train, 100)

    logits = torch.cat((view_1, view_2))
    logits_bis = torch.cat((view_2, view_1))

    labels = torch.randint(0, 100, (CONFIG.batch_size_train,))
    is_lab = torch.rand(CONFIG.batch_size_train) > 0.5

    cls = loss_classification(logits, labels, is_lab,
                              CONFIG.tau_s, CONFIG.tau_t,
                              CONFIG.epsilon)
    cls_bis = loss_classification(logits_bis, labels, is_lab,
                                  CONFIG.tau_s, CONFIG.tau_t,
                                  CONFIG.epsilon)

    assert np.isclose(cls['L_cls_u'].item(), cls_bis['L_cls_u'].item())
    assert np.isclose(cls['L_cls_s'].item(), cls_bis['L_cls_s'].item())

def test_hardcoded():
    logits = torch.Tensor([[1.0, 0.0], [0.0, 1.0]])
    labels = torch.zeros(1, dtype=torch.int64)
    is_lab = torch.tensor([True])

    tau_s, tau_t = 0.5, 0.25
    epsilon = 1.0
    cls = loss_classification(logits, labels, is_lab,
                              tau_s, tau_t,
                              epsilon)

    assert np.isclose(cls['logs']['cls/distill'].item(), 2.09095)
