import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy import signal
from scipy.io import wavfile

class Spectrogram:
    def __init__(self, frequencies, times, spectrogram):
        """
        create Spectrogram instance.
        """
        self.frequencies = frequencies
        self.times = times
        self.spectrogram = spectrogram

    def compute_matrix(self):
        # should I normalize my self.spectrogram here???
        pass

    def visualize(self):
        """
        plot the spectrogram.
        """
        # matplotlib pcolormesh function is used to plot the spectrogram. As an input it takes
        # coordinates of the quadrilateral corners of the mesh. Please refer to the matplotlib documentation for details.

        # corners_freq and corners_times store corner values for frequencies and times respectively
        corners_freq = np.hstack((0, 0.5 * self.frequencies[0:-1:1] + 0.5 * self.frequencies[1::], self.frequencies[-1]))
        corners_times = np.insert(self.times, 0, 0)

        plot_times, plot_freq = np.meshgrid(corners_times, corners_freq, sparse=False)

        fig = plt.figure(figsize=(7, 7))
        ax0 = fig.add_subplot(111)

        # using logarithmic scale for amplitudes
        im = ax0.pcolormesh(plot_times, plot_freq, np.log(self.spectrogram))
        fig.colorbar(im, ax=ax0)
        plt.show()

    def restore_recording(self, filename):
        """
        restore recording from the spectrogram and export it to .wav file.
        :param filename: name of .wav file to export recording
        """
        audio_signal = librosa.core.spectrum.griffinlim(self.spectrogram)
        # the next two lines just retreive sample rate from the Spectrogram fields.
        # do not know yet what 224 really is

        dt = self.times[1] - self.times[0]
        sample_rate = int(224 / dt)
               
        wavfile.write(filename, sample_rate, np.array(audio_signal, dtype=np.int16))

class Recording:
    def __init__(self, filename, n_channels=1, use_first=True):
        """
        create a Recording instance from file.
        :param filename: name of .wav file with recording
        """
        if n_channels == 1:
            self.sample_rate, self.samples = wavfile.read(filename)
        else:
            self.sample_rate, samples = wavfile.read(filename)
            self.samples = samples[:, 0 if use_first else 1]

    def compute_spectrogram(self):
        """
        compute a spectrogram for recording and create an instance
        of Spectrogram class.
        """
        frequencies, times, spectrogram = signal.spectrogram(self.samples, self.sample_rate)
        s = Spectrogram(frequencies, times, spectrogram)
        return s

class NMF:
    # stores the matrix for factorization
    def __init__(self, V):
        """create a NMF instances to factorize a non-negative matrix V.
            :param V: FxN matrix to be factorized
        """
        self.V = V

    def factorize_IS(self, K, n_iter):
        F, N = self.V.shape

        # initializing W and H
        W = np.abs(np.random.randn(F, K)) + np.ones((F, K))
        H = np.abs(np.random.randn(K, N)) + np.ones((K, N))
        
        for i in range(n_iter):
            #Update
            H = H * ( (np.transpose(W) @ (np.power(W@H, -2) * self.V)) * np.power(np.transpose(W) @ np.power(W@H,-1), -1) )
            
            W = W * ( ((np.power(W@H,-2) * self.V) @ np.transpose(H)) * np.power(np.power(W@H,-1) @ np.transpose(H), -1) )
            
            #Normalization
            for k in range(K):
                norm_factor = np.linalg.norm(W[:, k])
                W[:, k] = W[:, k] / norm_factor
                H[k, :] = H[k, :] * norm_factor
            
        return W, H, W@H

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

    def factorize_EUC(self, K, n_iter):
        F, N = self.V.shape

        # initializing W and H
        W = np.abs(np.random.randn(F, K)) + np.ones((F, K))
        H = np.abs(np.random.randn(K, N)) + np.ones((K, N))
        
        for i in range(n_iter):
            #Update
            H = H * ( (np.transpose(W) @ self.V) * np.power(np.transpose(W) @ W@H + 1e-09, -1) )
            
            W = W * ( (self.V @ np.transpose(H)) * np.power(W@H @ np.transpose(H) + 1e-09, -1) )
            #1e-09 to avoid division by 0
            #Normalization
            for k in range(K):
                norm_factor = np.linalg.norm(W[:, k])
                W[:, k] = W[:, k] / norm_factor
                H[k, :] = H[k, :] * norm_factor
                
        return W, H, W@H


    def factorize_KL(self, K, n_iter):
        F, N = self.V.shape

        # initializing W and H
        W = np.abs(np.random.randn(F, K)) + np.ones((F, K))
        H = np.abs(np.random.randn(K, N)) + np.ones((K, N))
        
        for i in range(n_iter):
            H = H * ( (np.transpose(W) @ (np.power(W@H,-1) * self.V)) * np.power( np.transpose(W) @ np.ones((F,N)), -1) )
            
            W = W * ( ((np.power(W@H, -1) * self.V) @ np.transpose(H)) * np.power( np.ones((F,N)) @ np.transpose(H), -1) )
            
            #Normalization
            for k in range(K):
                norm_factor = np.linalg.norm(W[:, k])
                W[:, k] = W[:, k] / norm_factor
                H[k, :] = H[k, :] * norm_factor
            
        return W, H, W@H
            
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

