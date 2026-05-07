import numpy as np
import matplotlib.pyplot as plt


class PCA:
    def __init__(self, data):

        self.X = np.array(data)
        self.Xmean = np.mean(self.X, axis=0)
        self.Xstd = self.X.std(axis=0,ddof=1)

        Xc = (self.X - self.Xmean) / self.Xstd
        cov_matrix = np.cov(Xc, rowvar=False)

        eigvals, eigvecs = np.linalg.eigh(cov_matrix)

        idx = np.argsort(eigvals)[::-1]
        self.eigvals = eigvals[idx]
        self.eigvecs = eigvecs[:, idx]

    def show_cum_ratio(self):
        
        plt.plot(self.eigvals/sum(self.eigvals),label="eigenvalues")
        
        plt.plot([
            sum(self.eigvals[: index]) / sum(self.eigvals)
            for index in range(len(self.eigvals))
        ],label="cumulative ratio")

        plt.legend()
        plt.show()

    def get_cum_ratio(self,index):
        return sum(self.eigvals[: index]) / sum(self.eigvals)

    def get_eigenvalues(self):
        return self.eigvals

    def get_eigenvectors(self):
        return self.eigvecs

    def get_expand_variance_ratio(self):
        return self.eigvals / sum(self.eigvals)

    def get_T2(self,data,components_num):

        data_std = (data-self.Xmean)/self.Xstd
        score = np.dot(data_std , self.eigvecs[:,:components_num])

        T2 = np.sum( (score**2) / self.eigvals[:components_num])

        return T2

    def get_each_T2(self,data,eignvec_index):
        data_std = (data-self.Xmean)/self.Xstd
        score = np.dot(data_std , self.eigvecs[eignvec_index])
        T2 = (score[:, eignvec_index]**2) / self.eigvals[eignvec_index]
        return T2

    def get_Q(self,data,components_num):

        data_mean = data - self.Xmean
        data_std = data_mean / self.Xstd

        data_reconstructed = self.eigvecs[:,:components_num] @ (self.eigvecs[:,:components_num].T @ data_std.T)
        residual = data_std - data_reconstructed

        Q = np.sum(residual**2)

        return Q

    def show_T2_report(self,data,components_num):
        data_std = (data-self.Xmean)/self.Xstd
        score = np.dot(data_std , self.eigvecs[:,:components_num])

        T2_vec =  (score**2) / self.eigvals[:components_num]

        plt.plot(T2_vec)
        plt.show()


    def show_Q_report(self,data,components_num):
        data_mean = data - self.Xmean
        data_std = data_mean / self.Xstd

        data_reconstructed = self.eigvecs[:,:components_num] @ (self.eigvecs[:,:components_num].T @ data_std.T)
        residual = data_std - data_reconstructed

        plt.plot(residual)
        plt.show()

    def get_T2_limit(self,components_num):

        data_num = len(self.X)
        index_99 = int((data_num * 0.95)//1 - 1)

        T2_values = [
            self.get_T2(item, components_num)
            for item in self.X
        ]

        T2_values.sort()

        return T2_values[index_99+1]


    def get_Q_limit(self,components_num):

        data_num = len(self.X)
        index_99 = int((data_num * 0.95)//1 -1)

        Q_values = [
            self.get_Q(item, components_num)
            for item in self.X
        ]

        Q_values.sort()

        return Q_values[index_99+1]