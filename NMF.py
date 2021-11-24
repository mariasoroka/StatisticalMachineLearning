import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
from scipy import signal
from scipy.io import wavfile


class Spectrogram:
    def __init__(self, abs_spectrogram, fs):
        """
        create Spectrogram instance.
        """
        self.spectrogram = abs_spectrogram
        self.fs = fs

    def compute_matrix(self):
        """
        compute the squared matrix of amplitudes.
        """
        return self.spectrogram ** 2

    def compute_frequencies_times(self):
        """
        restore frequencies and times values for spectrogram.
        """
        frequencies = librosa.fft_frequencies()
        times = librosa.frames_to_time(np.arange(np.shape(self.spectrogram)[1]))
        return frequencies, times

    def visualize(self):
        """
        plot the spectrogram.
        """
        fig, ax = plt.subplots()
        img = librosa.display.specshow(librosa.amplitude_to_db(self.spectrogram, ref=np.max), y_axis='log',
                                       x_axis='time', ax=ax)
        ax.set_title('Power spectrogram')
        fig.colorbar(img, ax=ax, format="%+2.0f dB")

    def restore_recording(self, filename):
        """
        restore recording from the spectrogram and export it to .wav file.
        :param filename: name of .wav file to export recording
        """
        audio_signal = librosa.core.spectrum.griffinlim(self.spectrogram)
        sf.write(filename, audio_signal, self.fs)


class Recording:
    def __init__(self, filename):
        """
        create a Recording instance from file.
        :param filename: name of .wav file with recording
        """
        self.sig, self.fs = librosa.core.load(filename, mono=True)

    def compute_spectrogram(self):
        """
        compute a spectrogram for recording and create an instance
        of Spectrogram class.
        """
        abs_spectrogram = np.abs(librosa.core.spectrum.stft(self.sig))
        s = Spectrogram(abs_spectrogram, self.fs)

        return s


class NMF:
    # stores the matrix for factorization
    def __init__(self, V):
        """create a NMF instances to factorize a non-negative matrix V.
            :param V: FxN matrix to be factorized
        """
        self.V = V
        self.costs = []
       
    def factorize_MU_IS(self, K, n_iter):
        """Factorize V ="""
        F, N = self.V.shape
        
        # initializing W and H
        W = np.abs(np.random.randn(F, K)) + np.ones((F, K))
        H = np.abs(np.random.randn(K, N)) + np.ones((K, N))
        
        WH = W@H

        self.costs = []
        self.costs.append(self.cost_divergence(WH, 1))
        
        WH_1 = np.power(WH,-1)
        WH_2 = np.power(WH,-2)
        
        for i in range(n_iter):
            H = H * ( (np.transpose(W) @ (WH_2 * self.V)) / (np.transpose(W) @ WH_1) ) + 1e-09
                
            WH = W@H
            WH_1 = np.power(WH,-1)
            WH_2 = np.power(WH, -2)
                
            W = W * ( ((WH_2 * self.V) @ np.transpose(H)) / (WH_1 @ np.transpose(H)) ) + 1e-09
                
            for k in range(K):
                norm_factor = np.linalg.norm(W[:, k])
                W[:, k] = W[:, k] / norm_factor
                H[k, :] = H[k, :] * norm_factor
                    
            WH = W@H
            WH_1 = np.power(WH,-1)
            WH_2 = np.power(WH, -2)
                
            self.costs.append(self.cost_divergence(WH, 0))
        return W, H, WH
        
    def factorize_EM_IS(self, K, n_iter, threshold=1E-10):
        """factorizes V in W @ H using the IS divergence following the EM algorithm.
            :param K: components size, V is a FxN matrix factorized into W and H,
            FxK and KxN matrices respectively
            :param n_iter: maximum number of iteration of the algorithm
            :param threshold: in order to prevent approximation error that could
            lead to negative value, under the threshold the update is 0
            :return: W, H, W @ H, FxK, KxN, FxN matrices s.t. V ~= W @ H
        """
        F, N = self.V.shape

        # initializing W and H
        W = np.abs(np.random.randn(F, K) + np.ones((F, K)))
        H = np.abs(np.random.randn(K, N) + np.ones((K, N)))

        WH = W @ H

        self.costs = []
        self.costs.append(self.cost_divergence(WH, 0))

        for i in range(n_iter):
            for k in range(K):  # SUGGESTION : shuffling range(K)
                old_w_k = W[:, k]
                old_h_k = H[k, :]
                wh = old_w_k[:, np.newaxis] @ old_h_k[np.newaxis, :]

                G_k = wh / WH  # Wiener gain
                V_k = np.power(G_k, 2) * self.V + (1 - G_k) * wh  # posterior power of C_k

                # updating column w_k and row h_k
                new_h_k = (np.power(old_w_k, -1).T @ V_k) / F
                new_w_k = (V_k @ np.power(new_h_k, -1).T) / N

                # normalisation (setting l2 norm of w_k to 1)
                norm_factor = np.linalg.norm(new_w_k)
                new_w_k = new_w_k / norm_factor
                new_h_k = new_h_k * norm_factor

                new_wh = new_w_k[:, np.newaxis] @ new_h_k[np.newaxis, :]

                # updating W, H and W @ H
                W[:, k] = new_w_k
                H[k, :] = new_h_k

                delta = new_wh - wh
                delta[np.abs(delta) < threshold] = 0
                WH = WH + delta

            self.costs.append(self.cost_divergence(WH, 0))

        return W, H, WH

    def factorize_EUC(self, K, n_iter):
        F, N = self.V.shape

        # initializing W and H
        W = np.abs(np.random.randn(F, K)) + np.ones((F, K))
        H = np.abs(np.random.randn(K, N)) + np.ones((K, N))

        WH = W @ H

        self.costs = []
        self.costs.append(self.cost_divergence(WH, 2))

        for i in range(n_iter):
            
            #Updates
            H = H * ( (np.transpose(W) @ self.V) * np.power(np.transpose(W) @ WH + 1e-09, -1) )

            WH = W @ H

            W = W * ( (self.V @ np.transpose(H)) * np.power(WH @ np.transpose(H) + 1e-09, -1) )
            #1e-09 to avoid division by 0
            
            #Normalization
            for k in range(K):
                norm_factor = np.linalg.norm(W[:, k])
                W[:, k] = W[:, k] / norm_factor
                H[k, :] = H[k, :] * norm_factor

            WH = W @ H

            self.costs.append(self.cost_divergence(WH, 2))

        return W, H, WH

    def factorize_KL(self, K, n_iter):
        F, N = self.V.shape

        # initializing W and H
        W = np.abs(np.random.randn(F, K)) + np.ones((F, K))
        H = np.abs(np.random.randn(K, N)) + np.ones((K, N))
        
        WH = W@H

        self.costs = []
        self.costs.append(self.cost_divergence(WH, 1))

        WH_1 = np.power(WH, -1)
        for i in range(n_iter):
            #Update of H
            H = H * ( (np.transpose(W) @ (WH_1 * self.V)) / (np.transpose(W) @ np.ones((F,N))) ) + 1e-09
            
            WH = W@H
            WH_1 = np.power(WH, -1)
            
            W = W * ( ( (WH_1 * self.V) @ np.transpose(H)) / (np.ones((F,N)) @ np.transpose(H)) ) + 1e-09  
            
            #Normalisation
            for k in range(K):         
               norm_factor = np.linalg.norm(W[:, k])
               W[:, k] = W[:, k] / norm_factor
               H[k, :] = H[k, :] * norm_factor
            
            WH = W@H
            WH_1 = np.power(WH, -1)

            self.costs.append(self.cost_divergence(WH, 1))
                        
        return W, H, W@H

    def factorize_R_EM_IS(self, K, n_iter, alpha, inverse_gamma=False, threshold=1E-10):
        """factorizes V in W @ H using the IS divergence with (inverse) Gamma Markov
            Chain prior to enforce smoothness, following the EM algorithm.
            :param K: components size, V is a FxN matrix factorized into W and H,
            FxK and KxN matrices respectively
            :param n_iter: maximum number of iteration of the algorithm
            :param alpha: regularization coefficient
            :param inverse_gamma: use inverse-Gamma Markov Chain (by default false)
            :param threshold: in order to prevent approximation error that could
            lead to negative value, under the threshold the update is 0
            :return: W, H, W @ H, FxK, KxN, FxN matrices s.t. V ~= W @ H
        """
        F, N = self.V.shape

        # initializing W and H
        W = np.abs(np.random.randn(F, K) + np.ones((F, K)))
        H = np.abs(np.random.randn(K, N) + np.ones((K, N)))

        WH = W @ H

        p_1 = np.repeat(F + 1, N)
        p_1[0] += -alpha if inverse_gamma else alpha
        p_1[N - 1] += alpha if inverse_gamma else -alpha

        index_boundary = N - 1 if inverse_gamma else 0

        b_p_2 = np.repeat(alpha + 1, N) if inverse_gamma else np.repeat(alpha - 1, N)
        b_p_2[index_boundary] = 1  # in order to not divide by 0, the boundaries are updated independently

        self.costs = []
        self.costs.append(self.cost_divergence(WH, 0))

        for i in range(n_iter):
            for k in range(K):  # SUGGESTION : shuffling range(K)
                old_w_k = W[:, k]
                old_h_k = H[k, :]
                wh = old_w_k[:, np.newaxis] @ old_h_k[np.newaxis, :]

                G_k = wh / WH  # Wiener gain
                V_k = np.power(G_k, 2) * self.V + (1 - G_k) * wh  # posterior power of C_k

                # updating column w_k and row h_k
                ML_h_k = (np.power(old_w_k, -1).T @ V_k) / F

                p_2 = b_p_2 / np.concatenate((old_h_k[1:], 1), axis=None) if inverse_gamma else b_p_2 / np.concatenate(
                    (1, old_h_k[0:-1]), axis=None)
                p_0 = -F * ML_h_k
                p_0 -= (alpha + 1) * np.concatenate((0, old_h_k[0:-1]), axis=None) if inverse_gamma else (
                                                                                                                 alpha - 1) * np.concatenate(
                    (old_h_k[1:], 0), axis=None)

                new_h_k = (np.sqrt((p_1 ** 2) - (4 * p_2 * p_0)) - p_1) / (2 * p_2)
                new_h_k[index_boundary] = -p_0[index_boundary] / p_1[index_boundary]
                new_w_k = (V_k @ np.power(new_h_k, -1).T) / N

                # normalisation (setting l2 norm of w_k to 1)
                norm_factor = np.linalg.norm(new_w_k)
                new_w_k = new_w_k / norm_factor
                new_h_k = new_h_k * norm_factor

                new_wh = new_w_k[:, np.newaxis] @ new_h_k[np.newaxis, :]

                # updating W, H and W @ H
                W[:, k] = new_w_k
                H[k, :] = new_h_k

                delta = new_wh - wh
                delta[np.abs(delta) < threshold] = 0
                WH = WH + delta

            self.costs.append(self.cost_divergence(WH, 0))

        return W, H, WH

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
            C = np.copy(X) / WH
            C = C * np.tile((W[:, k])[:, np.newaxis], N) * np.tile(H[k, :], (F, 1))

            C_matrices.append(C)

        return C_matrices

    def divergence(self, x, y, beta):
        """compute the beta divergence d_beta(x|y) of x from y.
        :param x: element whose divergence from y is computed
        :param y: reference element
        :param beta: control which divergence is chosen
        :return: d_beta(x|y)
        """
        if beta == 0:
            return x / (y+1e-09) - (np.log(x+1e-09) - np.log(y+1e-09)) - 1
        elif beta == 1:
            return x * (np.log(x+1e-09) - np.log(y + 1e-09)) + y - x
        else:
            return ((x ** beta) + (beta - 1) * (y ** beta) - beta * x * (y ** (beta - 1))) / (beta * (beta - 1))

    def cost_divergence(self, WH, beta):
        """compute the cost function based on the sum of beta divergences
            of WH from V.
        :param WH: matrix whose divergence from V is computed
        :param beta: control which divergence is chosen
        :return: d_beta(WH||V)
        """
        d = self.divergence(WH, self.V, beta)
        return np.sum(d)

    def plot_costs(self, method_name, beta):
        """plot the log(cost) vs iteration of the factorization procedure.
        :param method_name: name of the method used
        :param beta: beta-divergence used
        :return: None, show plot
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, title=r'$\beta$-divergence cost for {}'.format(method_name))
        ax.semilogy(range(len(self.costs)), self.costs, label=r'${}$-divergence'.format(beta))
        ax.set_xlabel('iteration')
        ax.set_ylabel('cost')
        ax.legend()
        plt.show()

    def summary_plot(self, K, n_iteration, threshold=1E-10):
        """compute the factorization of all algorithm and plot
            their log(cost) vs iteration.
        :param K: components size
        :param n_iteration: number of iteration used by algorithms
        :return: None, show plot
        """
        fig = plt.figure(figsize=(12, 12))

        self.factorize_EUC(K, n_iteration)
        ax1 = fig.add_subplot(221, title='EUC')
        ax1.semilogy(range(len(self.costs)), self.costs, label=r'${}$-divergence'.format(2))
        ax1.set_xlabel('iteration')
        ax1.set_ylabel('cost')
        ax1.legend()

        self.factorize_KL(K, n_iteration)
        ax2 = fig.add_subplot(222, title='KL')
        ax2.semilogy(range(len(self.costs)), self.costs, label=r'${}$-divergence'.format(1))
        ax2.set_xlabel('iteration')
        ax2.set_ylabel('cost')
        ax2.legend()

        self.factorize_MU_IS(K, n_iteration)
        ax3 = fig.add_subplot(223, title='IS/MU')
        ax3.semilogy(range(len(self.costs)), self.costs, label=r'${}$-divergence'.format(0))
        ax3.set_xlabel('iteration')
        ax3.set_ylabel('cost')
        ax3.legend()

        self.factorize_EM_IS(K, n_iteration, threshold=threshold)
        ax4 = fig.add_subplot(224, title='IS/EM')
        ax4.semilogy(range(len(self.costs)), self.costs, label=r'${}$-divergence'.format(0))
        ax4.set_xlabel('iteration')
        ax4.set_ylabel('cost')
        ax4.legend()

        plt.show()
