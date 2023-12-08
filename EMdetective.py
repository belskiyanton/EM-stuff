import numpy as np
from scipy.special import xlogy
from scipy.stats import gamma

def find_conv_trick(X, mask):
    '''
    X: 2D-array, shape(H, W)
    mask: 2D-array, shape(h, w)

    return: ans - 2D-array, shape(H - h + 1, W - w + 1)
    ans[i, j] = \sum_{k1 = 0}^{h - 1} \sum_{k2 = 0}^{w - 1} (mask[k1, k2] * X[k1 + i, k2 + j])

    это сумма выражения (mask[k1, k2] * X[k1 + i, k2 + j]) по k1, k2
    '''
    H = X.shape[0]
    W = X.shape[1]
    h = mask.shape[0]
    w = mask.shape[1]

    tmp = np.concatenate([mask[::-1, ::-1], np.zeros(shape=(H - h, w) + mask.shape[2:])], axis=0) #mask mirroring and expansion
    z = np.concatenate([tmp, np.zeros(shape=(H, W - w) + mask.shape[2:])], axis=1)

    tmp = np.zeros(shape=X.shape)
    for j in range(W):
        j1 = j + w - 1
        if j1 >= W:
            j1 = j1 - W
        for i in range(H):
            i1 = i + h - 1
            if i1 >= H:
                i1 = i1 - H
            tmp[i, j] = X[i1, j1] #some matrix shifting

    tmp = np.fft.fft(np.fft.fft(tmp, axis=0), axis=1)
    z = np.fft.fft(np.fft.fft(z, axis=0), axis=1) #2d fft but for other axes
    
    tmp = (tmp.T * z.T).T #some multiplication trick

    tmp = np.real(np.fft.ifft(np.fft.ifft(tmp, axis=1), axis=0)) #2d ifft but for other axes

    return tmp[:H - h + 1, :W - w + 1] #usefull data only

def special_log(x):
    return np.log(x, where=(x!=0))


def add_last_axis(X, K):
    return np.tile(X[:, :, np.newaxis], (1, 1, K))

def sum_sqr_diff_along_face(X, F, B):
    H = X.shape[0]
    W = X.shape[1]
    K = X.shape[2]
    h = F.shape[0]
    w = F.shape[1]

    base = (F * F).sum()
    
    Bk = add_last_axis(B, K)
    dot = Bk * (2 * X - Bk)
    dot_sum = dot.cumsum(axis=0).cumsum(axis=1)
    
    ans = [[] for i in range(H - h + 1)]
    for i in range(H - h + 1):
        for j in range(W - w + 1):
            dot_22 = dot_sum[i + h - 1, j + w - 1] # Inclusion–exclusion principle will be used
            dot_12 = 0
            dot_21 = 0
            dot_11 = 0
            if (i > 0):
                dot_12 = dot_sum[i - 1, j + w - 1]
            if (j > 0):
                dot_21 = dot_sum[i + h - 1, j - 1]
            if (i > 0) and (j > 0):
                dot_11 = dot_sum[i - 1, j - 1]

            sum_dot_along_face = dot_22 - dot_21 - dot_12 + dot_11
            ans[i] += [sum_dot_along_face]
    ans = np.array(ans)
    
    ans = ans + base
    
    ans = ans - 2 * find_conv_trick(X, F)

    return ans

def sum_diff_sqr_first2axes(X, B):
    # tmp = X - add_last_axis(B, X.shape[2])
    # return (tmp * tmp).sum(axis=(0, 1))

    tmp = (B.T - X.T)
    return (tmp * tmp).sum(axis=(1, 2))

def sum_sqr_diff_along_face_map(X, F, B, q, im):
    h = F.shape[0]
    w = F.shape[1]

    i = q[0, im]
    j = q[1, im]
    b_under_face = B[i:i + h, j:j + w]
    tmp = (b_under_face + F) - 2 * X[i:i + h, j:j + w, im] 
    tmp = tmp * (F - b_under_face)
    return tmp.sum(axis=(0, 1))

def my_softmax2d(z):
    y = z - z.max(axis=(0, 1))
    y = np.exp(y)
    y = y / y.sum(axis=(0, 1))
    return y

def find_map(z):
    ind = z.reshape(z.shape[0] * z.shape[1], z.shape[2]).argmax(axis=0)
    return np.array([ind // z.shape[1], ind % z.shape[1]])

def calculate_log_probability(X, F, B, s):
    """
    Calculates log p(X_k|d_k,F,B,s) for all images X_k in X and
    all possible displacements d_k.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    F : array, shape (h, w)
        Estimate of villain's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.

    Returns
    -------
    ll : array, shape(H-h+1, W-w+1, K)
        ll[dh,dw,k] - log-likelihood of observing image X_k given
        that the villain's face F is located at displacement (dh, dw)
    """
    H = X.shape[0]
    W = X.shape[1]
    K = X.shape[2]
    h = F.shape[0]
    w = F.shape[1]
    
    base = -(H * W * np.log(2 * np.pi * s * s)) / 2 #first constant

    base = base - (sum_diff_sqr_first2axes(X, B) / (2 * s * s)) #second constant but not for different pictures
    
    ans = (-1/(2 * s * s)) * sum_sqr_diff_along_face(X, F, B)
    
    ans += base
    return ans

def calculate_lower_bound(X, F, B, s, A, q, use_MAP=False):
    """
    Calculates the lower bound L(q,F,B,s,A) for the marginal log likelihood.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    F : array, shape (h, w)
        Estimate of villain's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    q : array
        If use_MAP = False: shape (H-h+1, W-w+1, K)
            q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
            of villain's face given image Xk
        If use_MAP = True: shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    use_MAP : bool, optional
        If true then q is a MAP estimates of displacement (dh,dw) of
        villain's face given image Xk.

    Returns
    -------
    L : float
        The lower bound L(q,F,B,s,A) for the marginal log likelihood.
    """
    H = X.shape[0]
    W = X.shape[1]
    K = X.shape[2]
    if use_MAP:
        ans = -(H * W * K * np.log(2 * np.pi * s * s)) / 2 - (sum_diff_sqr_first2axes(X, B).sum() / (2 * s * s))

        addition = 0
        for im in range(K):
            addition += sum_sqr_diff_along_face_map(X, F, B, q, im)
        ans += (-1 / (2 * s * s)) * addition
        
        ans += np.log(A[q[0], q[1]]).sum()
        return ans

    return (q * calculate_log_probability(X, F, B, s)).sum() + (q.sum(axis=2) * special_log(A)).sum() - xlogy(q, q).sum()


def run_e_step(X, F, B, s, A, use_MAP=False):
    """
    Given the current esitmate of the parameters, for each image Xk
    esitmates the probability p(d_k|X_k,F,B,s,A).

    Parameters
    ----------
    X : array, shape(H, W, K)
        K images of size H x W.
    F  : array_like, shape(h, w)
        Estimate of villain's face.
    B : array shape(H, W)
        Estimate of background.
    s : scalar, shape(1, 1)
        Eestimate of standard deviation of Gaussian noise.
    A : array, shape(H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    use_MAP : bool, optional,
        If true then q is a MAP estimates of displacement (dh,dw) of
        villain's face given image Xk.

    Returns
    -------
    q : array
        If use_MAP = False: shape (H-h+1, W-w+1, K)
            q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
            of villain's face given image Xk
        If use_MAP = True: shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    """
    H = X.shape[0]
    W = X.shape[1]
    K = X.shape[2]
    h = F.shape[0]
    w = F.shape[1]

    ans = (-1/(2 * s * s)) * sum_sqr_diff_along_face(X, F, B) #logits
    
    y = ans - ans.max(axis=(0, 1))
    y = np.exp(y)
    y = y * add_last_axis(A, K)
    if use_MAP:
        return find_map(y)
    else:
        return y / y.sum(axis=(0, 1))


def run_m_step(X, q, h, w, use_MAP=False, prior_smoothing=0):
    """
    Estimates F,B,s,A given esitmate of posteriors defined by q.

    Parameters
    ----------
    X : array, shape(H, W, K)
        K images of size H x W.
    q  :
        if use_MAP = False: array, shape (H-h+1, W-w+1, K)
           q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
           of villain's face given image Xk
        if use_MAP = True: array, shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    h : int
        Face mask height.
    w : int
        Face mask width.
    use_MAP : bool, optional
        If true then q is a MAP estimates of displacement (dh,dw) of
        villain's face given image Xk.

    Returns
    -------
    F : array, shape (h, w)
        Estimate of villain's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    """
    H = X.shape[0]
    W = X.shape[1]
    K = X.shape[2]

    if use_MAP:
        q1 = np.zeros(shape=(H - h + 1, W - w + 1)) + (prior_smoothing / ((H - h + 1) * (W - w + 1)))
        for im in range(K):
            i = q[0, im]
            j = q[1, im]
            q1[i, j] += 1
        A = q1 / (K + prior_smoothing)
        
        F = np.zeros(shape=(h, w))
        for im in range(K):
            i = q[0, im]
            j = q[1, im]
            F += X[i:i+h, j:j+w, im]
        F = F / K
        
        B = X.sum(axis=2)
        P = np.ones(shape=(H, W)) * K
        for im in range(K):
            i = q[0, im]
            j = q[1, im]
            B[i:i + h, j:j + w] -= X[i:i + h, j:j + w, im]
            P[i:i + h, j:j + w] -= np.ones(shape=(h, w))
#         B = B / P
        B = np.divide(B, P, where=(P!=0))
        
        C = sum_diff_sqr_first2axes(X, B).sum()
        for im in range(K):
            C += sum_sqr_diff_along_face_map(X, F, B, q, im)
        s = np.sqrt(C / (K * H * W))
        return F, B, s, A
    else:
        A = q.mean(axis=2)
        
        F = find_conv_trick(X, q).mean(axis=2)
        
        q_sum = q.cumsum(axis=0).cumsum(axis=1)

        B = [[] for k1 in range(H)]
        for k1 in range(H):
            for k2 in range(W):
                i1 = max(0, k1 + 1 - h)
                j1 = max(0, k2 + 1 - w)
                i2 = min(k1, q.shape[0] - 1)
                j2 = min(k2, q.shape[1] - 1)
                
                # i in [i1, i2]; j in [j1, j2]; 
                # q_xy = q_sum[i_x, j_y], there i_2 = i2, i_1 = i1 - 1; but if it is out of range, then zero
                
                q_22 = q_sum[i2, j2] # Inclusion–exclusion principle will be used
                q_12 = 0
                q_21 = 0
                q_11 = 0
                if (i1 > 0):
                    q_12 = q_sum[i1 - 1, j2]
                if (j1 > 0):
                    q_21 = q_sum[i2, j1 - 1]
                if (i1 > 0) and (j1 > 0):
                    q_11 = q_sum[i1 - 1, j1 - 1]
                
                sum_q_along_face = q_22 - q_21 - q_12 + q_11
                
                sum_q_not_face = 1 - sum_q_along_face
                
                B[k1] += [(X[k1, k2] * sum_q_not_face).sum() / sum_q_not_face.sum()]
        B = np.array(B)

        C = sum_diff_sqr_first2axes(X, B).sum()
        C += (q * sum_sqr_diff_along_face(X, F, B)).sum()

        s = np.sqrt(C / (K * H * W))
        
        return F, B, s, A

def run_EM(X, h, w, F=None, B=None, s=None, A=None, tolerance=0.001,
           max_iter=50, use_MAP=False, prior_smoothing=0, calculate_ELBO=True):
    """
    Runs EM loop until the likelihood of observing X given current
    estimate of parameters is idempotent as defined by a fixed
    tolerance.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    h : int
        Face mask height.
    w : int
        Face mask width.
    F : array, shape (h, w), optional
        Initial estimate of villain's face.
    B : array, shape (H, W), optional
        Initial estimate of background.
    s : float, optional
        Initial estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1), optional
        Initial estimate of prior on displacement of face in any image.
    tolerance : float, optional
        Parameter for stopping criterion.
    max_iter  : int, optional
        Maximum number of iterations.
    use_MAP : bool, optional
        If true then after E-step we take only MAP estimates of displacement
        (dh,dw) of villain's face given image Xk.

    Returns
    -------
    F, B, s, A : trained parameters.
    LL : array, shape(number_of_iters,)
        L(q,F,B,s,A) after each EM iteration (1 iteration = 1 e-step + 1 m-step); 
        number_of_iters is actual number of iterations that was done.
    """
    H = X.shape[0]
    W = X.shape[1]
    K = X.shape[2]
    
    if (F is None):
        F = np.zeros(shape=(h, w))
    if (s is None):
        s = 1
    if (B is None):
        B = np.zeros(shape=(H, W))
    if (A is None):
        #A = np.ones(shape=(H - h + 1, W - w + 1)) / ((H - h + 1) * (W - w + 1))
        eps = 1e-5
        A = np.ones(shape=(H - h + 1, W - w + 1)) * eps
        A[0, 0] = 1 - (((H - h + 1) * (W - w + 1)) - 1) * eps
    hist = []
    l_old = -np.inf
    for k in range(max_iter):
        q = run_e_step(X, F, B, s, A, use_MAP=use_MAP)
        F, B, s, A = run_m_step(X, q, h, w, use_MAP=use_MAP, prior_smoothing=prior_smoothing)
        if calculate_ELBO or (k % 5 == 0):
            l_now = calculate_lower_bound(X, F, B, s, A, q, use_MAP=use_MAP)
            hist += [l_now]

            if (l_now - l_old) < tolerance:
                break
            else:
                l_old = l_now
    hist = np.array(hist)
    return F, B, s, A, hist

def run_EM_with_restarts(X, h, w, tolerance=0.001, max_iter=50, use_MAP=False,
                         n_restarts=10, prior_smoothing=0, calculate_ELBO=True):
    """
    Restarts EM several times from different random initializations
    and stores the best estimate of the parameters as measured by
    the L(q,F,B,s,A).

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    h : int
        Face mask height.
    w : int
        Face mask width.
    tolerance, max_iter, use_MAP : optional parameters for EM.
    n_restarts : int
        Number of EM runs.

    Returns
    -------
    F : array,  shape (h, w)
        The best estimate of villain's face.
    B : array, shape (H, W)
        The best estimate of background.
    s : float
        The best estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1)
        The best estimate of prior on displacement of face in any image.
    L : float
        The best L(q,F,B,s,A).
    """
    H = X.shape[0]
    W = X.shape[1]
    K = X.shape[2]
    
    best_score = -np.inf
    for n in range(n_restarts):
        A = my_softmax2d(np.random.rand(H - h + 1, W - w + 1))
        B = np.random.rand(H, W)
        F = np.random.rand(h, w)
        s = gamma.rvs(1, size=1)[0]
        F, B, s, A, hist = run_EM(X, h, w, F=F, B=B, s=s, A=A, tolerance=tolerance,
           max_iter=max_iter, use_MAP=use_MAP, prior_smoothing=prior_smoothing, calculate_ELBO=calculate_ELBO)
        if hist[-1] > best_score:
            best_score = hist[-1]
            best_F = F
            best_B = B
            best_s = s
            best_A = A
    return best_F, best_B, best_s, best_A, best_score