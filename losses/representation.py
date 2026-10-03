import torch
import torch.nn.functional as F

def contrastive_loss(features, positive_mask, tau):
    """Parameters
    features: Tensor(N, D) float32 the projected output of the head
    positive_mask: Tensor(N, N) bool which terms to take into account
    tau: float temperature

    Returns
    loss: torch scalar the contrastive loss

    The i-th row of positive_mask gives the other positive views
    of the i-th sample. The loss is calculated by averaging the terms
    over each anchor, and then averaging over the anchors.
    This matches both the supervised and unsupervised loss formula.
    """
    N = features.shape[0]
    device = features.device

    features = F.normalize(features, dim=1)
    logits = features @ features.T / tau # (N, N)
    logits[torch.eye(N, dtype=torch.bool, device=device)] = -torch.inf
    assert logits.shape == (N, N)

    log_prob = F.log_softmax(logits, dim=1)
    masked_log_prob = torch.where(positive_mask, log_prob, 0.0)

    loss_per_anchor = -1.0 * masked_log_prob.sum(dim=1) / positive_mask.sum(dim=1) # (N,)
    loss = loss_per_anchor.mean(dim=0) # scalar

    return loss

def loss_representation(features, labels, is_lab, tau_u, tau_c):
    """Parameters
    features: Tensor(2B, 256) float32 the projected output of the head
    labels: Tensor(B,) int64 the labels
    is_lab: Tensor(B,) bool is the sample labelled
    tau_u, tau_c: float unsupervised and supervised temperatures

    Returns
    {
        'L_rep_u': torch.float32,
        'L_rep_s': torch.float32
    }
    """
    B = features.shape[0] // 2
    assert len(labels) == B
    assert len(is_lab) == B
    device = features.device

    ### unsupervised
    mask = torch.eye(2*B, dtype=torch.bool, device=device).roll(B, dims=0)
    L_rep_u = contrastive_loss(features, mask, tau_u)

    ### supervised
    features = features[torch.cat((is_lab, is_lab))] # (2B_l, 256)
    labels = labels[is_lab] # (B_l, )
    B_l = labels.shape[0]

    labels_twice = torch.cat((labels, labels))
    is_same_label = (
        torch.unsqueeze(labels_twice, 0) == torch.unsqueeze(labels_twice, 1)
    )
    not_self = ~torch.eye(2*B_l, dtype=torch.bool, device=device)
    mask = not_self & is_same_label

    L_rep_s = contrastive_loss(features, mask, tau_c)

    return {
        'L_rep_u': L_rep_u,
        'L_rep_s': L_rep_s
    }
