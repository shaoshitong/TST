def correct_num(output, target, topk=(1,)):
    """
    compute the top1 and top5
    """
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    if target.shape != output.shape:
        correct = pred.eq(target.view(1, -1).expand_as(pred))
    else:
        correct = pred.eq(target.argmax(1).view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
