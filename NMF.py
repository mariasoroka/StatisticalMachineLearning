import numpy as np

class Spectrogram:
    def __init__(self):
        pass
    def compute_matrix(self):
        pass
    def visualize(self):
        pass

class Recording:
    def __init__(self, filename):
        pass
    def compute_spectrogram(self):
        pass

class NMF:
    # stores the matrix for factorization
    def __init__(self, V):
        """create a NMF instances to factorize a non-negative matrix V.
            :param V: FxN matrix to be factorized
        """
        self.V = V

    def factorize_IS(self):
        pass

    def factorize_EM_IS(self, K, n_iter):
        """factorizes V in W @ H using the IS divergence following the EM algorithm.
            :param K: components size, V is a FxN matrix factorized into W and H,
            FxK and KxN matrices respectively
            :param n_iter: maximum number of iteration of the algorithm
            :return: W, H, W @ H, FxK, KxN, FxN matrices s.t. V ~= W @ H
        """
        F, N = self.V.shape

        # initializing W and H
        W = np.abs(np.random.randn(F, K) + np.ones((F, K)))
        H = np.abs(np.random.randn(K, N) + np.ones((K, N)))

        WH = W @ H

        for i in range(n_iter):
            for k in range(K): # SUGGESTION : shuffling range(K)
                old_w_k = W[:, k]
                old_h_k = H[k, :]
                wh = old_w_k[:, np.newaxis] @ old_h_k[np.newaxis, :]

                G_k = wh/WH # Wiener gain
                V_k = np.power(G_k, 2)*self.V + (1-G_k)*wh # posterior power of C_k

                # updating column w_k and row h_k
                new_w_k = (V_k @ np.power(old_h_k, -1).T)/N
                new_h_k = (np.power(old_w_k, -1).T @ V_k)/F

                # normalisation (setting l2 norm of w_k to 1)
                norm_factor = np.linalg.norm(new_w_k)
                new_w_k = new_w_k/norm_factor
                new_h_k = new_h_k * norm_factor

                new_wh = new_w_k[:, np.newaxis] @ new_h_k[np.newaxis, :]

                # updating W, H and W @ H
                W[:, k] = new_w_k
                H[k, :] = new_h_k
                WH = WH - wh + new_wh

        return W, H, WH

    def factorize_EUC(self):
        pass

    def factorize_KL(self):
        pass

    def wiener_reconstruction(self, W, H, WH=None):
        """reconstruct the components when seeing np.sqrt(V) = sum of gaussian components (frame dependent),
            using V NMF factorization.
            :param W: W as obtained by factorization
            :param H: H as obtained by factorization
            :param WH: W @ H if computed, else it is computed again (optional)
            :return: list of matrices for every components, corresponding to the contribution in np.sqrt(V) of
            each component (i.e. np.sqrt(V) = sum of those matrices)
        """
        # recomputing W @ H if not provided
        if WH is None:
            WH = W @ H

        X = np.sqrt(self.V)
        F, N = X.shape
        K = W.shape[1]

        C_matrices = []
        for k in range(K):
            C = np.copy(X)/WH
            C = C * np.tile((W[:, k])[:, np.newaxis], N) * np.tile(H[k, :], (F, 1))

            C_matrices.append(C)

        return C_matrices

