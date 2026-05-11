import numpy as np
import matplotlib.pyplot as plt


class KernelPCA:
    def __init__(self, data, n_components,kernel="rbf", gamma=1):

        self.gamma = gamma

        self.X = np.array(data)
        self.Xmean = np.mean(self.X, axis=0)
        self.Xstd = self.X.std(axis=0,ddof=1)

        self.Xc = (self.X - self.Xmean) / self.Xstd

        self.N = len(self.Xc)

        self.K_matrix = np.array([[self.centered_kernel_function(xi, xj) for xj in self.Xc] for xi in self.Xc])

        self.eigen_values, self.eigen_vectors = np.linalg.eig(self.K_matrix)


    def centered_kernel_function(self, x, y):

        kxy = self.k_function(x, y)
        kxxc = np.mean([self.k_function(x, xi) for xi in self.Xc])
        kyxc = np.mean([self.k_function(xi, y) for xi in self.Xc])
        kxcxc = np.mean([self.k_function(xi, xj) for xi in self.Xc for xj in self.Xc])

        return kxy - kxxc - kyxc + kxcxc

    def k_function(self,x,y):
        return np.exp(-self.gamma * np.sum((np.array(x) - np.array(y)) ** 2))

    def alpha_vectors(self):
        return np.array([alpha / np.sqrt(self.N * lambda_val) for alpha, lambda_val in zip(self.eigen_vectors.T, self.eigen_values)])

    def lambda_values(self):
        return self.eigen_values / self.N

    def score(self,data,eigenvec_index):
        data_c = (data - self.Xmean) / self.Xstd

        alpha_vec = self.alpha_vectors()[eigenvec_index]
        kernel_vec = np.array([self.centered_kernel_function(data_c, xi) for xi in self.Xc   ])

        return np.dot(alpha_vec, kernel_vec)

    def show_cum_ratio(self):
        plt.plot(self.eigen_values/sum(self.eigen_values),label="eigenvalues")
        
        plt.plot([
            sum(self.eigen_values[: index]) / sum(self.eigen_values)
            for index in range(len(self.eigen_values))
        ],label="cumulative ratio")

        plt.legend()
        plt.show()

    def get_cum_ratio(self,index):
        return sum(self.eigen_values[: index]) / sum(self.eigen_values)

    def get_expand_variance_ratio(self):
        #TODO 未実装 
        return self.eigvals / sum(self.eigvals)

    def get_T2(self,data,components_num):
        scores = [self.score(data,index)**2/self.lambda_values()[index] for index in range(components_num)]

        print("scores",[self.score(data,index) for index in range(components_num)])
        print("lambda_values",[self.lambda_values()[index] for index in range(components_num)])


        return sum(scores)


    def get_Q_2(self,data,components_num):

        pass

    def get_each_T2(self,data,eignvec_index):
        data_c = (data - self.Xmean) / self.Xstd
        Z = self.kpca.transform(data_c.reshape(1, -1))[:,eignvec_index]
        mu = self.eigvals[eignvec_index]

        T2 = np.sum((Z**2) / mu)
        return T2

    def get_Q(self,data,components_num):
        #TODO 未実装
        data_c = (data - self.Xmean) / self.Xstd
        print("data_c",data_c)
        print("data_c",data_c.reshape(1, -1))
        print(self.kpca._get_kernel(data_c.reshape(1, -1)))
        print(self.kpca._get_kernel(self.Xc))

        Q = 0

        #data_reconstructed = self.eigvecs[:,:components_num] @ (self.eigvecs[:,:components_num].T @ data_std.T)
        #residual = data_std - data_reconstructed

        #Q = np.sum(residual**2)

        return Q


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