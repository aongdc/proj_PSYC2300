import numpy as np

rng = np.random.default_rng()
mems = rng.choice([-1, 1], size=(500, 400))
pairs = np.array([mems[:,:-1:2], mems[:,1::2]]).reshape((1000, -1))
N = pairs.shape[0]

# pairs = np.array([[-1, 1, 1], [1, 1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1], [1, -1, 1]])
# pairs = np.array([[-1, 1], [1, 1], [1, -1], [-1, 1], [1, 1], [1, -1]])
# train
W = np.zeros((N, N))
for pair in pairs[None].T:
    W += pair @ pair.T 
np.fill_diagonal(W, 0)

# add noise
xi = rng.choice([-1, 1], p=[0.2, 0.8], size=(500, 200))
noisy_pairs = np.array([mems[:,:-1:2], mems[:, 1::2] * xi]).reshape((N, -1))
# probe=noisy_pairs.T[0]
# update
# probe = np.array([1, 1, 0, 0, 0, 0])
success = dict()
failure = []
for k, probe in enumerate(noisy_pairs.T):
    print(k)
    s = np.copy(probe)
    succ = False
    num_iter = 20000
    i = 0
    while i < num_iter:
        i += 1
        prev_s = np.copy(s)
        idx = rng.integers(499, N)
        s[idx] = np.sign(W[idx] @ s)
        if np.equal(s, pairs.T).all(1).sum() > 0:
            print(i)
            print(np.equal(s, pairs.T).all(1).nonzero())
            succ = True
            break
    if succ:
        success[k] = i
    else:
        failure.append(k)
        # if np.array_equal(prev_s, s):
        #     print(i, 'same')
        #     break
        # s = np.sign(W @ s)

np.equal(s, pairs.T).sum(axis=1)/len(pairs)*100

def energy(s):
    return -0.5 * s @ W @ s + np.sum(s * threshold)

print(success)