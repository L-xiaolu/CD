import paddle

def std_mean(input, dim=None, unbiased=True, keepdim=False):
    std = paddle.std(input, axis=dim,
                     unbiased=unbiased, keepdim=keepdim)
    mean = paddle.mean(input,
                       axis=dim,
                       keepdim=keepdim)
    return std, mean