import numpy as np

def qrs_correct(result):
    pos = np.argwhere(result > 0.5).flatten()
    diff_pos = np.diff(pos)

    rpos = []
    pre = 0
    last = len(pos)-1
    cand_poses = np.where(diff_pos >= 15)[0]
    if last > 1:
        for index, cand_pos in enumerate(cand_poses):
            if cand_pos - pre >= 15:
                r_cand = round((pos[pre] + pos[cand_pos]) / 2)
                # print (r_cand)
                rpos.append(r_cand)
            pre = cand_pos + 1

        rpos.append(round((pos[pre] + pos[last]) / 2))
        qrs = np.array(rpos)
        qrs = qrs.astype(int)
    else:
        qrs = np.array([])

    qrs_diff = np.diff(qrs)
    check = True
    r = 0
    while check:
        qrs_diff = np.diff(qrs)
        if len(qrs_diff[qrs_diff <= 50]) == 0:
            check = False
            break
        while r < len(qrs_diff):
            if qrs_diff[r] <= 50:
                prev = qrs[r]
                next = qrs[r+1]
                qrs = np.delete(qrs,r)
                qrs = np.delete(qrs,r)
                qrs = np.insert(qrs, r, (prev+next)/2)
                qrs_diff = np.diff(qrs)
            r += 1
        check = False

    return qrs

def hs_correct(preds):
    V = np.zeros_like(preds)
    Idx = np.zeros_like(preds)

    eps = 1e-6
    lnP = np.log(preds + eps)
    #lnP = preds
    V[0] = lnP[0]
    next = np.array([1, 2, 3, 0])
    prev = np.array([3, 0, 1, 2])
    for i in range(1, preds.shape[0]):
        for j in range(preds.shape[1]):
            #s = V[i-1, j] + lnP[i, j]
            #p = V[i-1, prev[j]] + max(lnP[i, j], lnP[i, (j+2)%4])
            s = V[i-1, j] + lnP[i, j]
            p = V[i-1, prev[j]] + lnP[i, j]

            if s > p:
                V[i, j] = s
                Idx[i, j] = j
            else:
                V[i, j] = p
                Idx[i, j] = prev[j]

    opt_path = np.zeros((preds.shape[0], ))
    opt_path[-1] = np.argmax(V[-1, :])
    for i in range(preds.shape[0]-2, 0, -1):
        opt_path[i] = Idx[i+1, int(opt_path[i+1])]

    return opt_path
