import torch
import torch.nn.functional as F

def teacher_temp(epoch, config):
    """Returns tau_t for the current epoch indice"""
    if epoch >= config.warmup_epochs:
        return config.tau_t
    return config.warmup_tau_t + float(epoch/(config.warmup_epochs-1))*(config.tau_t - config.warmup_tau_t)

def loss_classification(logits, labels, is_lab,
                        tau_s, tau_t,
                        epsilon):
    """Parameters
    logits: Tensor (2B, 100) float32 the cosine logits out of the head
    labels: Tensor (B, ) int64 the labels
    is_lab: Tensor (B, ) bool is labelled
    tau_s, tau_t: float the temperatures of student and teacher (for the current epoch)
    epsilon: float weights the entropy regularizer

    Returns
    {
        'L_cls_u': torch.float32,
        'L_cls_s': torch.float32,
        'logs': {'cls/entropy': torch.float32,
                 'cls/distill': torch.float32}
    }
    """
    B = logits.shape[0] // 2
    assert len(labels) == B

    student_logits = logits
    teacher_logits = logits.detach()

    student_logits = 1/tau_s * student_logits
    teacher_logits = 1/tau_t * teacher_logits

    # entropy
    p_bar = torch.mean(F.softmax(student_logits, dim=1), dim=0)
    H = -torch.sum(p_bar * torch.log(p_bar))

    # supervised
    input_lab = student_logits[torch.cat((is_lab, is_lab))]
    target = torch.cat((labels[is_lab], labels[is_lab]))
    L_cls_s = F.cross_entropy(input_lab, target,
                              reduction='mean')
    if not torch.any(is_lab): # in the rare case of 0 labelled sample
        L_cls_s = logits.new_zeros(()) # we return a zero on the same device to avoid an error
    
    # unsupervised
    rolled_views = torch.roll(teacher_logits, B, 0)
    distill = F.cross_entropy(student_logits, F.softmax(rolled_views, dim=1),
                         reduction='mean')
    L_cls_u = distill - epsilon*H

    return {
        'L_cls_u': L_cls_u,
        'L_cls_s': L_cls_s,
        'logs': {'cls/entropy': H.detach(),
                 'cls/distill': distill.detach()}
    }
