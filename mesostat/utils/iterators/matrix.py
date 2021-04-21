
# Return a list of all pairs of elements in a list, excluding flips.
def iter_g_2D(lst):
    n = len(lst)
    assert n >= 2
    for i in range(n):
        for j in range(i + 1, n):
            yield [lst[i], lst[j]]


def iter_gg_3D(lst):
    n = len(lst)
    assert n >= 3
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                yield [lst[i], lst[j], lst[k]]


def iter_gn_3D(lst):
    n = len(lst)
    assert n >= 3
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(n):
                if (k != i) and (k != j):
                    yield [lst[i], lst[j], lst[k]]