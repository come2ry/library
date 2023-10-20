
# https://atcoder.jp/contests/abc231/submissions/27890257
def compress(*iters):
    s = set()
    for iter in iters:
        s = s.union(set(iter))
    unique = sorted(s)
    comp = {x: i for i, x in enumerate(unique)}
    ret = []
    for iter in iters:
        ret.append([comp[x] for x in iter])
    return unique, comp, ret


# https://atcoder.jp/contests/abc231/submissions/28104147
def zahyo_compress(lst):
    XS = sorted(set(lst))
    return {e: i for i, e in enumerate(XS)}
