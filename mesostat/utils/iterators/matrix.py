

# Return a list of all pairs of elements in a list, excluding flips.
def iter_g_2D(n):
    assert n >= 2
    for i in range(n):
        for j in range(i + 1, n):
            yield i, j


def iter_gg_3D(n):
    assert n >= 3
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                yield i, j, k


def iter_gn_3D(n):
    assert n >= 3
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(n):
                if (k != i) and (k != j):
                    yield i, j, k


def iter_ggn_4D(n):
    assert n >= 4
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(n):
                    if (l != i) and (l != j) and (l != k):
                        yield i, j, k, l
