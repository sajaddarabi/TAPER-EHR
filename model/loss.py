import torch
import torch.nn as nn
import torch.nn.functional as F

def mse_loss(output, target, i_s):
    mse = torch.nn.MSELoss(size_average=True)
    loss = 0.0
    i_s = i_s + 1
    for i in range(target.shape[1]):
        if (len(i_s.shape) == 0):
            n = i_s.item()
        else:
            n = i_s[i].item()
        for j in range(n ):
            loss += mse(output[j, i], target[j, i])
    loss = (1.0 / (float(torch.sum(i_s)))) * loss
    if (loss == 0.0):
        import pdb; pdb.set_trace()
        loss = torch.tensor(0.0, device=inputs.device)
    return loss

def nll_loss(output, target):
    output = torch.squeeze(output)
    target = torch.squeeze(target)
    return F.nll_loss(output, target)

def multiclass_loss(output, target):
    bce_loss = nn.BCEWithLogitsLoss()
    return bce_loss(output, target.float())

def skipgram_loss_neg(u_hat, v, positive_val):
    # [batch_size, neg_size, emb_dim] x [batch_size, emb_dim, 1] = [batch_size, neg_size, 1]
    # neg_vals: [batch_size, neg_size]

    neg_vals = torch.bmm(u_hat, v.unsqueeze(2)).squeeze(2)
    # neg_val: [batch_size]
    neg_val = F.logsigmoid(-torch.sum(neg_vals, dim=1)).squeeze()
    loss = positive_val + neg_val
    return -loss.mean()


def bce_loss(output, target, weight=[1.0, 1.0]):
    weight = ((target == 1).float() * weight[1]) + ((target == 0).float() * weight[0])
    loss = nn.BCELoss(weight)
    #target = target.float()
    #return - (target * output.log() + (1-target) * (1 - output).log()).mean()
    #loss = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1.0, 7.0]).to(output.device))
    #output = torch.unsqueeze(output, dim=1)
    #target = torch.unsqueeze(target, dim=1)
    return loss(output, target.float())

def med2vec_loss_transformer(inputs, mask, probits, bce_loss, emb_w, ivec, jvec, window=1, eps=1.0e-8):
    """ returns the med2vec loss
    """
    def visit_loss(x, mask, probits, window=1):
        loss = 0
        mask = mask.bool()
        for i in range(0, window):

            if loss != loss:
                import pdb; pdb.set_trace()
            if (i == 0):
                maski = mask[i + 1:] * mask[:-i - 1]
            else:
                maski = mask[i + 1:] * mask[1:-i] * mask[:-i - 1]

            forward_preds = torch.masked_select(probits[:-i-1], maski) #* maski
            backward_preds = torch.masked_select(probits[i+1:], maski)
            forward_xi = torch.masked_select(x[i+1:].float(), maski)
            backward_xi = torch.masked_select(x[:-i-1].float(), maski)
            tl = bce_loss(forward_preds, forward_xi) + bce_loss(backward_preds, backward_xi)
            if (not torch.isnan(tl)):
                loss += tl
        # BUG: for certain batches loss is infinite.. issue with mask??
        if loss == 0 or torch.isnan(loss):
            loss = torch.tensor(0.0, device=inputs.device)
        return loss

    def code_loss(emb_w, ivec, jvec, eps=1.e-6):
        norm = torch.sum(torch.exp(torch.mm(emb_w.t(), emb_w)), dim=1)

        cost = -torch.log((torch.exp(torch.sum(emb_w[:, ivec].t() * emb_w[:, jvec].t(), dim=1)) / norm[ivec]) + eps)
        cost = torch.mean(cost)
        return cost

    vl = visit_loss(inputs, mask, probits, window=window)
    cl = code_loss(emb_w, ivec, jvec, eps=1.e-6)
    return {'visit_loss': vl, 'code_loss': cl}

def med2vec_loss(inputs, mask, probits, bce_loss, emb_w, ivec, jvec, window=1, eps=1.0e-8):
    """ returns the med2vec loss
    """
    def visit_loss(x, mask, probits, window=1):
        loss = 0
        for i in range(0, window):
            if loss != loss:
                import pdb; pdb.set_trace()
            if (i == 0):
                maski = mask[i + 1:] * mask[:-i - 1]
            else:
                maski = mask[i + 1:] * mask[1:-i] * mask[:-i - 1]
            backward_preds = probits[i+1:] * maski
            forward_preds = probits[:-i-1] * maski
            tl = bce_loss(forward_preds, x[i+1:].float()) + bce_loss(backward_preds, x[:-i-1].float())
            if (not torch.isnan(tl)):
                loss += tl

        return loss

    def code_loss(emb_w, ivec, jvec, eps=1.e-6):
        norm = torch.sum(torch.exp(torch.mm(emb_w.t(), emb_w)), dim=1)

        cost = -torch.log((torch.exp(torch.sum(emb_w[:, ivec].t() * emb_w[:, jvec].t(), dim=1)) / norm[ivec]) + eps)
        cost = torch.mean(cost)
        return cost

    vl = visit_loss(inputs, mask, probits, window=window)
    cl = code_loss(emb_w, ivec, jvec, eps=1.e-6)
    return {'visit_loss': vl, 'code_loss': cl}
