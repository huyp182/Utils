import numpy as np

def dropout(x, keep_prob, training=True):
    if training:
        ref = np.random.binomial(1, keep_prob, x.shape)
        x = x * ref
        x = x / keep_prob
    else:
        x = x * keep_prob

    return x


if __name__ == '__main__':
    x = np.ones((3, 4))
    print(dropout(x, keep_prob=0.9, training=True))
