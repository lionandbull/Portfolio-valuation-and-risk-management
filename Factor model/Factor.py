import numpy as np
import pandas as pd
import datetime
import csv
import matplotlib.pyplot as plt
from efficient_frontier import efficientFrontier
from prettytable import PrettyTable

class solver:
    def __init__(self, risk_free, *files):
        self.risk_free = risk_free
        self.files = files
        self.price_list = []
        self.expected_return = []
        self.returns = []
        self.mu_vector = []
        self.Omega = []
        self.omega_bar = 0
        self.omega_T = 0
        self.mu_f = 0
        self.names = []
        self.beta = []
        self.alpha = []
        self.train = []
        self.test = []
        self.ff_factors_list = []
        self.ff_factors_train = []
        self.ff_factors_test = []
        self.momentum_list = []
        self.momentum_train = []
        self.momentum_test = []
        self.exchange_list = []
        self.exchange_train = []
        self.exchange_test = []
        self.date = []
        self.Sigma_F = []
        self.expected_return = []
        self.return_variance = []
        self.covariance = []
        self.cov_matrix = []
        self.models = ['CAPM_F_1', 'CAPM_F_2', 'CAPM_F_1_F_2', 'FF_F_1', 'FF_F_2', 'FF_F_1_F_2']
        self.price_test = []
        self.model_expected_return = []




    def get_FF_factors(self, file):
        csv = pd.read_csv(file)
        csv.drop(['Mkt-RF', 'RF'], axis=1, inplace=True)
        self.ff_factors_list = [item[1:] for item in csv.as_matrix()]
        self.ff_factors_list = [item.tolist() for item in self.ff_factors_list]
        self.ff_factors_list = np.asarray(self.ff_factors_list)
        #print(self.ff_factors_list)
        date = [i[0] for i in csv.as_matrix()]
        date = np.asarray(date)
        index_start = np.nonzero(date == 20160301)[0][0]
        index_end = np.nonzero(date == 20161230)[0][0]
        self.ff_factors_train = self.ff_factors_list[index_start:index_end+1]
        self.ff_factors_train = self.ff_factors_train[1:]
        self.ff_factors_test = self.ff_factors_list[index_end+1:]
        #for i in range(len(self.test)):
         #   print(self.test[i], ": ", self.date[i])




    def get_momentume(self, file):
        csv = pd.read_csv(file)
        self.momentum_list = [item[1:][0] for item in csv.as_matrix()]
        self.momentum_list = np.asarray(self.momentum_list)
        #print(self.momentum_list)
        date = [i[0] for i in csv.as_matrix()]
        date = np.asarray(date)
        index_start = np.nonzero(date == 20160301)[0][0]
        index_end = np.nonzero(date == 20161230)[0][0]
        #print(date[index_end:index_end + 1])
        #print(date[index_end + 1])
        #print(date[-1])
        self.momentum_train = self.momentum_list[index_start:index_end + 1]
        self.momentum_train = self.momentum_train[1:]
        self.momentum_test = self.momentum_list[index_end + 1:]




    def get_exchange(self, file):
        csv = pd.read_csv(file)
        csv.drop(['PX_LAST', '% Change'], axis=1, inplace=True)
        date = [i[0] for i in csv.as_matrix()]
        date.reverse()
        date = np.asarray(date)
        self.exchange_list = [item[1:][0] for item in csv.as_matrix()]
        self.exchange_list.reverse()
        self.exchange_list = np.asarray(self.exchange_list)
        self.exchange_list = np.float64(self.exchange_list)
        #print(self.exchange_list)
        index = np.nonzero(date == '12/30/16')[0][0]
        #print(index)
        #print(self.exchange_list[index+1])
        self.exchange_train = self.exchange_list[0:index + 1]
        self.exchange_train = self.exchange_train[1:]
        self.exchange_test = self.exchange_list[index + 1:]
        self.date = date




    def get_price_list(self, years):
        index = years/3
        csv = pd.read_csv(self.files[0])
        date = [i[0] for i in csv.as_matrix()]
        date = np.asarray(date)

        #for i in range(len(date)):
        #    date[i] = datetime.datetime.strptime(date[i], '%Y-%m-%d').strftime('%-m/%-d/%y')

        position = []
        for i in range(len(date)):
            if date[i] not in self.date:
                position.append(i)
        date_new = []
        for i in range(len(date)):
            if i not in position:
                date_new.append(date[i])
            else:
                continue
        date = date_new

        for file in self.files:
            csv = pd.read_csv(file)
            csv.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
            list = []
            for item in csv.as_matrix():
                list.append(item[1])
            list_new = []
            for i in range(len(list)):
                if i not in position:
                    list_new.append(list[i])
                else:
                    continue
            self.price_list.append(list_new)

        for i in self.price_list:
            returns = []
            for j in range(len(i) - 1):
                #new_return = (i[j + 1] - i[j]) / i[j]
                new_return = ((i[j + 1] - i[j]) / i[j])
                new_return = np.log(1 + new_return)
                returns.append(new_return)
            self.returns.append(np.asarray(returns))
        self.returns = self.annulize(self.returns)
        index = np.nonzero(np.asarray(date) == '12/30/16')[0][0]
        self.train = [sub[:index] for sub in self.returns]
        self.test = [sub[index:] for sub in self.returns]
        ##-0.00838804807107, 0.00284521463789

        self.price_test = [sub[index+1:] for sub in self.price_list[0:-1]]

    def annulize(self, array):
        array = np.asarray(array)
        return array*210
        #return (1 + array)**(252/210) - 1

    def threeMatrixDot(self, a, b, c):
        return np.dot( np.dot(a, b), c)[0][0]

    ## Construct Sigma_F
    def cal_Sigma_F(self):
        market_factor = [item for item in (self.train[-1] - self.risk_free)]
        sml = [item[0] for item in self.ff_factors_train]
        hml = [item[1] for item in self.ff_factors_train]
        momentum = [item for item in self.momentum_train]
        exchange = [item for item in self.exchange_train]
        self.Sigma_F = np.cov([market_factor, sml, hml, momentum, exchange])

    def capm_traditional(self, factors):
        risk_free = self.risk_free
        self.beta = []
        self.alpha = []
        if factors == 0:
            for i in range(len(self.returns[0:-1])):
                R = self.train[i] - risk_free
                ones = np.ones([len(R), 1])
                X = np.copy(self.train[-1])
                X = X - risk_free
                X.shape = (len(X), 1)
                X = np.concatenate((ones, X), axis=1)
                result = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)), R)
                self.beta.append(result[1:])
                self.alpha.append(result[0])
            self.beta = np.asarray(self.beta)
            self.alpha = np.asarray(self.alpha)

        elif factors == 'CAPM_F_1':
            for i in range(len(self.returns[0:-1])):
                R = self.train[i] - risk_free
                #R = self.annulize(self.train[i]) - risk_free
                ones = np.ones([len(R), 1])
                X = np.copy(self.train[-1])

                #X = self.annulize(X)

                X = X - risk_free
                X.shape = (len(X), 1)
                X = np.concatenate((ones, X), axis=1)
                momentum = [[i] for i in self.momentum_train]
                X = np.concatenate((X, momentum), axis=1)
                result = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)), R)
                self.beta.append(result[1:])
                self.alpha.append(result[0])
            self.beta = np.asarray(self.beta)
            self.alpha = np.asarray(self.alpha)
            expected_return = []
            return_variance = []
            covariance = []
            for i in range(len(self.returns[0:-1])):
                expected_return.append(self.risk_free + self.alpha[i] + self.beta[i][0] * np.mean(self.train[-1] - self.risk_free)
                                       + self.beta[i][1] * np.mean(self.momentum_train))
                returns = self.risk_free + self.alpha[i] + self.beta[i][0] * (self.train[-1] - self.risk_free) + \
                          self.beta[i][1] * self.momentum_train
                residual = returns - self.train[i]

                residual = np.transpose([residual])
                Sigma_F = np.array([self.Sigma_F[i] for i in [0, 3]])
                Sigma_F = Sigma_F.T
                Sigma_F = np.array([Sigma_F[i] for i in [0, 3]])
                beta = np.array([item for item in self.beta[i]])
                beta = np.transpose([beta])
                return_variance.append(self.threeMatrixDot(np.transpose(beta), Sigma_F, beta)
                                       + (np.dot(np.transpose(residual), residual)
                                       / (len(self.train[0]) - len(self.beta[0]) - 1))[0][0])

                temp_cov = []
                for j in range(i+1,len(self.train[0:-1])):
                    beta_i = np.array([item for item in self.beta[i]])
                    beta_i = np.transpose([beta_i])
                    beta_j = np.array([item for item in self.beta[j]])
                    beta_j = np.transpose([beta_j])
                    temp_cov.append(self.threeMatrixDot(np.transpose(beta_i), Sigma_F, beta_j))
                covariance.append(temp_cov)

            self.expected_return = expected_return
            self.return_variance = return_variance
            self.covariance = covariance
            cov_matrix = []
            for i in range(len(self.return_variance)):
                temp = [0 for j in range(i)]
                temp.append(self.return_variance[i])
                temp = temp + self.covariance[i]
                cov_matrix.append(temp)
            cov_matrix = np.asarray(cov_matrix)
            self.cov_matrix = cov_matrix
            transpose = np.transpose(self.cov_matrix)
            for i in range(len(self.cov_matrix)):
                for j in range(len(self.cov_matrix[0])):
                    if j < i:
                        self.cov_matrix[i][j] = transpose[i][j]
            self.return_variance = np.asarray(self.return_variance)/ 210
            self.cov_matrix = self.cov_matrix/210




        elif factors == 'CAPM_F_2':
            for i in range(len(self.returns[0:-1])):
                R = self.train[i] - risk_free
                ones = np.ones([len(R), 1])
                X = np.copy(self.train[-1])
                X = X - risk_free
                X.shape = (len(X), 1)
                X = np.concatenate((ones, X), axis=1)
                exchange = [[i] for i in self.exchange_train]
                X = np.concatenate((X, exchange), axis=1)
                result = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)), R)
                self.beta.append(result[1:])
                self.alpha.append(result[0])
            self.beta = np.asarray(self.beta)
            self.alpha = np.asarray(self.alpha)

            expected_return = []
            return_variance = []
            covariance = []
            for i in range(len(self.returns[0:-1])):
                expected_return.append(self.risk_free + self.alpha[i] + self.beta[i][0] * np.mean(self.train[-1] - self.risk_free)
                                       + self.beta[i][1] * np.mean(self.exchange_train))
                returns = self.risk_free + self.alpha[i] + self.beta[i][0] * (self.train[-1] - self.risk_free) + \
                          self.beta[i][1] * self.exchange_train
                residual = returns - self.train[i]
                residual = np.transpose([residual])
                Sigma_F = np.array([self.Sigma_F[i] for i in [0, 4]])
                Sigma_F = Sigma_F.T
                Sigma_F = np.array([Sigma_F[i] for i in [0, 4]])
                beta = np.array([item for item in self.beta[i]])
                beta = np.transpose([beta])
                return_variance.append(self.threeMatrixDot(np.transpose(beta), Sigma_F, beta)
                                       + (np.dot(np.transpose(residual), residual) / (
                                               len(self.train[0]) - len(self.beta[0]) - 1))[0][0])

                temp_cov = []
                for j in range(i + 1, len(self.train[0:-1])):
                    beta_i = np.array([item for item in self.beta[i]])
                    beta_i = np.transpose([beta_i])
                    beta_j = np.array([item for item in self.beta[j]])
                    beta_j = np.transpose([beta_j])
                    temp_cov.append(self.threeMatrixDot(np.transpose(beta_i), Sigma_F, beta_j))
                covariance.append(temp_cov)
            self.expected_return = expected_return
            self.return_variance = return_variance
            self.covariance = covariance
            cov_matrix = []
            for i in range(len(self.return_variance)):
                temp = [0 for j in range(i)]
                temp.append(self.return_variance[i])
                temp = temp + self.covariance[i]
                cov_matrix.append(temp)
            cov_matrix = np.asarray(cov_matrix)
            self.cov_matrix = cov_matrix
            transpose = np.transpose(self.cov_matrix)
            for i in range(len(self.cov_matrix)):
                for j in range(len(self.cov_matrix[0])):
                    if j < i:
                        self.cov_matrix[i][j] = transpose[i][j]
            self.return_variance = np.asarray(self.return_variance) / 210
            self.cov_matrix = self.cov_matrix / 210

        elif factors == 'CAPM_F_1_F_2':
            for i in range(len(self.returns[0:-1])):
                R = self.train[i] - risk_free
                ones = np.ones([len(R), 1])
                X = np.copy(self.train[-1])
                X = X - risk_free
                X.shape = (len(X), 1)
                X = np.concatenate((ones, X), axis=1)
                exchange = [[i] for i in self.exchange_train]
                momentum = [[i] for i in self.momentum_train]
                X = np.concatenate((X, momentum, exchange), axis=1)
                result = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)), R)
                self.beta.append(result[1:])
                self.alpha.append(result[0])
            self.beta = np.asarray(self.beta)
            self.alpha = np.asarray(self.alpha)
            expected_return = []
            return_variance = []
            covariance = []
            for i in range(len(self.returns[0:-1])):
                expected_return.append(self.risk_free + self.alpha[i] + self.beta[i][0] * np.mean(self.train[-1] - self.risk_free)
                                       + self.beta[i][1] * np.mean(self.momentum_train)
                                       + self.beta[i][2] * np.mean(self.exchange_train))
                returns = self.risk_free + self.alpha[i] + self.beta[i][0] * (self.train[-1] - self.risk_free) + \
                          self.beta[i][1] * self.momentum_train + self.beta[i][2] * self.exchange_train
                residual = returns - self.train[i]
                residual = np.transpose([residual])
                Sigma_F = np.array([self.Sigma_F[i] for i in [0, 3, 4]])
                Sigma_F = Sigma_F.T
                Sigma_F = np.array([Sigma_F[i] for i in [0, 3, 4]])
                beta = np.array([item for item in self.beta[i]])
                beta = np.transpose([beta])
                return_variance.append(self.threeMatrixDot(np.transpose(beta), Sigma_F, beta)
                                       + (np.dot(np.transpose(residual), residual) / (
                                               len(self.train[0]) - len(self.beta[0]) - 1))[0][0])

                temp_cov = []
                for j in range(i + 1, len(self.train[0:-1])):
                    beta_i = np.array([item for item in self.beta[i]])
                    beta_i = np.transpose([beta_i])
                    beta_j = np.array([item for item in self.beta[j]])
                    beta_j = np.transpose([beta_j])
                    temp_cov.append(self.threeMatrixDot(np.transpose(beta_i), Sigma_F, beta_j))
                covariance.append(temp_cov)
            self.expected_return = expected_return
            self.return_variance = return_variance
            self.covariance = covariance
            cov_matrix = []
            for i in range(len(self.return_variance)):
                temp = [0 for j in range(i)]
                temp.append(self.return_variance[i])
                temp = temp + self.covariance[i]
                cov_matrix.append(temp)
            cov_matrix = np.asarray(cov_matrix)
            self.cov_matrix = cov_matrix
            transpose = np.transpose(self.cov_matrix)
            for i in range(len(self.cov_matrix)):
                for j in range(len(self.cov_matrix[0])):
                    if j < i:
                        self.cov_matrix[i][j] = transpose[i][j]
            self.return_variance = np.asarray(self.return_variance) / 210
            self.cov_matrix = self.cov_matrix / 210


    def FF(self, factors):
        risk_free = self.risk_free
        self.beta = []
        self.alpha = []
        if factors == 0:
            for i in range(len(self.returns[0:-1])):
                R = self.train[i] - risk_free
                ones = np.ones([len(R), 1])
                X = np.copy(self.train[-1])
                X = X - risk_free
                X.shape = (len(X), 1)
                X = np.concatenate((ones, X), axis=1)
                sml = [[item[0]] for item in self.ff_factors_train]
                sml = np.asarray(sml)
                hml = [[item[1]] for item in self.ff_factors_train]
                hml = np.asarray(hml)
                X = np.concatenate((X, sml, hml), axis=1)
                result = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)), R)
                self.beta.append(result[1:])
                self.alpha.append(result[0])
            self.beta = np.asarray(self.beta)
            self.alpha = np.asarray(self.alpha)

        if factors == 'FF_F_1':
            for i in range(len(self.returns[0:-1])):
                R = self.train[i] - risk_free
                ones = np.ones([len(R), 1])
                X = np.copy(self.train[-1])
                X = X - risk_free
                X.shape = (len(X), 1)
                X = np.concatenate((ones, X), axis=1)
                sml = [[item[0]] for item in self.ff_factors_train]
                sml = np.asarray(sml)
                hml = [[item[1]] for item in self.ff_factors_train]
                hml = np.asarray(hml)
                momentum = [[i] for i in self.momentum_train]
                X = np.concatenate((X, sml, hml, momentum), axis=1)
                result = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)), R)
                self.beta.append(result[1:])
                self.alpha.append(result[0])
            self.beta = np.asarray(self.beta)
            self.alpha = np.asarray(self.alpha)
            expected_return = []
            return_variance = []
            covariance = []
            for i in range(len(self.returns[0:-1])):
                expected_return.append(self.risk_free + self.alpha[i] + self.beta[i][0] * np.mean(self.train[-1] - self.risk_free)
                                       + self.beta[i][1] * np.mean(self.ff_factors_train[:, 0])
                                       + self.beta[i][2] * np.mean(self.ff_factors_train[:, 1])
                                       + self.beta[i][3] * np.mean(self.momentum_train))
                returns = self.risk_free + self.alpha[i] + self.beta[i][0] * (self.train[-1] - self.risk_free)\
                          + self.beta[i][1] * self.ff_factors_train[:, 0]\
                          + self.beta[i][2] * self.ff_factors_train[:, 1]\
                          + self.beta[i][3] * self.momentum_train
                residual = returns - self.train[i]
                residual = np.transpose([residual])
                Sigma_F = np.array([self.Sigma_F[i] for i in [0, 1, 2, 3]])
                Sigma_F = Sigma_F.T
                Sigma_F = np.array([Sigma_F[i] for i in [0, 1, 2, 3]])
                beta = np.array([item for item in self.beta[i]])
                beta = np.transpose([beta])
                return_variance.append(self.threeMatrixDot(np.transpose(beta), Sigma_F, beta)
                                       + (np.dot(np.transpose(residual), residual) / (
                                               len(self.train[0]) - len(self.beta[0]) - 1))[0][0])

                temp_cov = []
                for j in range(i + 1, len(self.train[0:-1])):
                    beta_i = np.array([item for item in self.beta[i]])
                    beta_i = np.transpose([beta_i])
                    beta_j = np.array([item for item in self.beta[j]])
                    beta_j = np.transpose([beta_j])
                    temp_cov.append(self.threeMatrixDot(np.transpose(beta_i), Sigma_F, beta_j))
                covariance.append(temp_cov)
            self.expected_return = expected_return
            self.return_variance = return_variance
            self.covariance = covariance
            cov_matrix = []
            for i in range(len(self.return_variance)):
                temp = [0 for j in range(i)]
                temp.append(self.return_variance[i])
                temp = temp + self.covariance[i]
                cov_matrix.append(temp)
            cov_matrix = np.asarray(cov_matrix)
            self.cov_matrix = cov_matrix
            transpose = np.transpose(self.cov_matrix)
            for i in range(len(self.cov_matrix)):
                for j in range(len(self.cov_matrix[0])):
                    if j < i:
                        self.cov_matrix[i][j] = transpose[i][j]
            self.return_variance = np.asarray(self.return_variance) / 210
            self.cov_matrix = self.cov_matrix / 210


        if factors == 'FF_F_2':
            for i in range(len(self.returns[0:-1])):
                R = self.train[i] - risk_free
                ones = np.ones([len(R), 1])
                X = np.copy(self.train[-1])
                X = X - risk_free
                X.shape = (len(X), 1)
                X = np.concatenate((ones, X), axis=1)
                sml = [[item[0]] for item in self.ff_factors_train]
                sml = np.asarray(sml)
                hml = [[item[1]] for item in self.ff_factors_train]
                hml = np.asarray(hml)
                exchange = [[i] for i in self.exchange_train]
                X = np.concatenate((X, sml, hml, exchange), axis=1)
                result = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)), R)
                self.beta.append(result[1:])
                self.alpha.append(result[0])
            self.beta = np.asarray(self.beta)
            self.alpha = np.asarray(self.alpha)
            expected_return = []
            return_variance = []
            covariance = []
            for i in range(len(self.returns[0:-1])):
                expected_return.append(self.risk_free + self.alpha[i] + self.beta[i][0] * np.mean(self.train[-1] - self.risk_free)
                                       + self.beta[i][1] * np.mean(self.ff_factors_train[:, 0])
                                       + self.beta[i][2] * np.mean(self.ff_factors_train[:, 1])
                                       + self.beta[i][3] * np.mean(self.exchange_train))
                returns = self.risk_free + self.alpha[i] + self.beta[i][0] * (self.train[-1] - self.risk_free)\
                          + self.beta[i][1] * self.ff_factors_train[:, 0]\
                          + self.beta[i][2] * self.ff_factors_train[:, 1]\
                          + self.beta[i][3] * self.exchange_train
                residual = returns - self.train[i]
                residual = np.transpose([residual])
                Sigma_F = np.array([self.Sigma_F[i] for i in [0, 1, 2, 4]])
                Sigma_F = Sigma_F.T
                Sigma_F = np.array([Sigma_F[i] for i in [0, 1, 2, 4]])
                beta = np.array([item for item in self.beta[i]])
                beta = np.transpose([beta])
                return_variance.append(self.threeMatrixDot(np.transpose(beta), Sigma_F, beta)
                                       + (np.dot(np.transpose(residual), residual) / (
                                               len(self.train[0]) - len(self.beta[0]) - 1))[0][0])

                temp_cov = []
                for j in range(i + 1, len(self.train[0:-1])):
                    beta_i = np.array([item for item in self.beta[i]])
                    beta_i = np.transpose([beta_i])
                    beta_j = np.array([item for item in self.beta[j]])
                    beta_j = np.transpose([beta_j])
                    temp_cov.append(self.threeMatrixDot(np.transpose(beta_i), Sigma_F, beta_j))
                covariance.append(temp_cov)
            self.expected_return = expected_return
            self.return_variance = return_variance
            self.covariance = covariance
            cov_matrix = []
            for i in range(len(self.return_variance)):
                temp = [0 for j in range(i)]
                temp.append(self.return_variance[i])
                temp = temp + self.covariance[i]
                cov_matrix.append(temp)
            cov_matrix = np.asarray(cov_matrix)
            self.cov_matrix = cov_matrix
            transpose = np.transpose(self.cov_matrix)
            for i in range(len(self.cov_matrix)):
                for j in range(len(self.cov_matrix[0])):
                    if j < i:
                        self.cov_matrix[i][j] = transpose[i][j]
            self.return_variance = np.asarray(self.return_variance) / 210
            self.cov_matrix = self.cov_matrix / 210

        if factors == 'FF_F_1_F_2':
            for i in range(len(self.returns[0:-1])):
                R = self.train[i] - risk_free
                ones = np.ones([len(R), 1])
                X = np.copy(self.train[-1])
                X = X - risk_free
                X.shape = (len(X), 1)
                X = np.concatenate((ones, X), axis=1)
                sml = [[item[0]] for item in self.ff_factors_train]
                sml = np.asarray(sml)
                hml = [[item[1]] for item in self.ff_factors_train]
                hml = np.asarray(hml)
                momentum = [[i] for i in self.momentum_train]
                exchange = [[i] for i in self.exchange_train]
                X = np.concatenate((X, sml, hml, momentum, exchange), axis=1)
                result = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)), R)
                self.beta.append(result[1:])
                self.alpha.append(result[0])
            self.beta = np.asarray(self.beta)
            self.alpha = np.asarray(self.alpha)
            expected_return = []
            return_variance = []
            covariance = []
            for i in range(len(self.returns[0:-1])):
                expected_return.append(self.risk_free + self.alpha[i] + self.beta[i][0] * np.mean(self.train[-1] - self.risk_free)
                                       + self.beta[i][1] * np.mean(self.ff_factors_train[:, 0])
                                       + self.beta[i][2] * np.mean(self.ff_factors_train[:, 1])
                                       + self.beta[i][3] * np.mean(self.momentum_train)
                                       + self.beta[i][4] * np.mean(self.exchange_train))
                returns = self.risk_free + self.alpha[i] + self.beta[i][0] * (self.train[-1] - self.risk_free)\
                          + self.beta[i][1] * self.ff_factors_train[:, 0]\
                          + self.beta[i][2] * self.ff_factors_train[:, 1]\
                          + self.beta[i][3] * self.momentum_train\
                          + self.beta[i][4] * self.exchange_train
                residual = returns - self.train[i]
                residual = np.transpose([residual])
                Sigma_F = self.Sigma_F
                beta = np.array([item for item in self.beta[i]])
                beta = np.transpose([beta])
                return_variance.append(self.threeMatrixDot(np.transpose(beta), Sigma_F, beta)
                                       + (np.dot(np.transpose(residual), residual) / (
                                               len(self.train[0]) - len(self.beta[0]) - 1))[0][0])

                temp_cov = []
                for j in range(i + 1, len(self.train[0:-1])):
                    beta_i = np.array([item for item in self.beta[i]])
                    beta_i = np.transpose([beta_i])
                    beta_j = np.array([item for item in self.beta[j]])
                    beta_j = np.transpose([beta_j])
                    temp_cov.append(self.threeMatrixDot(np.transpose(beta_i), Sigma_F, beta_j))
                covariance.append(temp_cov)
            self.expected_return = expected_return
            self.return_variance = return_variance
            self.covariance = covariance
            cov_matrix = []
            for i in range(len(self.return_variance)):
                temp = [0 for j in range(i)]
                temp.append(self.return_variance[i])
                temp = temp + self.covariance[i]
                cov_matrix.append(temp)
            cov_matrix = np.asarray(cov_matrix)
            self.cov_matrix = cov_matrix
            transpose = np.transpose(self.cov_matrix)
            for i in range(len(self.cov_matrix)):
                for j in range(len(self.cov_matrix[0])):
                    if j < i:
                        self.cov_matrix[i][j] = transpose[i][j]
            self.return_variance = np.asarray(self.return_variance) / 210
            self.cov_matrix = self.cov_matrix / 210

    def construct_models(self):
        name = ["AAL", "AAPL", "BABA", "BIDU", "BURL", "D", "DAL",
                "FB", "FDX", "FRT", "GNC", "GOOG", "GPRO", "L",
                "NKE", "O", "S", "SPWR", "T", "UPS"]
        table_beta = []

        for i in range(len(self.models)):
            if i == 0:
                x = PrettyTable(["Asset", 'Beta_0(intercept)', 'Beta_1(R_m - R_f)', 'Beta_2(Momentum)'])
                x.align["Asset"] = "l"
                x.padding_width = 1
                self.capm_traditional(self.models[i])
                for j in range(len(name)):
                    x.add_row([name[j]] + [str(self.alpha[j])] + [str(self.beta[j][0])] + [str(self.beta[j][1])])
                table_beta.append(x)

                Xaxis = np.arange(len(name))
                plt.barh(Xaxis, self.beta[:, 0])
                plt.yticks(Xaxis, name)
                plt.xlabel('Beta_1(R_m - R_f)')
                plt.title('Assets')
                plt.show()

                Xaxis = np.arange(len(name))
                plt.barh(Xaxis, self.beta[:, 1])
                plt.yticks(Xaxis, name)
                plt.xlabel('Beta_2(Momentum)')
                plt.title('Assets')
                plt.show()


            if i == 1:
                x = PrettyTable(["Asset", 'Beta_0(intercept)', 'Beta_1(R_m - R_f)', 'Beta_2(China_us_exchange)'])
                x.align["Asset"] = "l"
                x.padding_width = 1
                self.capm_traditional(self.models[i])
                for j in range(len(name)):
                    x.add_row([name[j]] + [str(self.alpha[j])] + [str(self.beta[j][0])] + [str(self.beta[j][1])])
                table_beta.append(x)

                Xaxis = np.arange(len(name))
                plt.barh(Xaxis, self.beta[:, 0])
                plt.yticks(Xaxis, name)
                plt.xlabel('Beta_1(R_m - R_f)')
                plt.title('Assets')
                plt.show()

                Xaxis = np.arange(len(name))
                plt.barh(Xaxis, self.beta[:, 1])
                plt.yticks(Xaxis, name)
                plt.xlabel('Beta_2(China_US_Exchange)')
                plt.title('Assets')
                plt.show()

            if i == 2:
                x = PrettyTable(["Asset", 'Beta_0(intercept)', 'Beta_1(R_m - R_f)', 'Beta_2(Momentum)', 'Beta_3(China_us_exchange)'])
                x.align["Asset"] = "l"
                x.padding_width = 1
                self.capm_traditional(self.models[i])
                for j in range(len(name)):
                    x.add_row([name[j]] + [str(self.alpha[j])] + [str(self.beta[j][0])] + [str(self.beta[j][1])] + [str(self.beta[j][2])])
                table_beta.append(x)

                Xaxis = np.arange(len(name))
                plt.barh(Xaxis, self.beta[:, 0])
                plt.yticks(Xaxis, name)
                plt.xlabel('Beta_1(R_m - R_f)')
                plt.title('Assets')
                plt.show()

                Xaxis = np.arange(len(name))
                plt.barh(Xaxis, self.beta[:, 1])
                plt.yticks(Xaxis, name)
                plt.xlabel('Beta_2(Momentum)')
                plt.title('Assets')
                plt.show()

                Xaxis = np.arange(len(name))
                plt.barh(Xaxis, self.beta[:, 2])
                plt.yticks(Xaxis, name)
                plt.xlabel('Beta_2(China_US_Exchange)')
                plt.title('Assets')
                plt.show()

            if i == 3:
                x = PrettyTable(["Asset", 'Beta_0(intercept)', 'Beta_1(R_m - R_f)', 'Beta_2(SML)', 'Beta_3(HML)', 'Beta_4(Momentum)'])
                x.align["Asset"] = "l"
                x.padding_width = 1
                self.FF(self.models[i])
                for j in range(len(name)):
                    x.add_row([name[j]] + [str(self.alpha[j])] + [str(self.beta[j][0])] + [str(self.beta[j][1])] +
                              [str(self.beta[j][2])] + [str(self.beta[j][3])])
                table_beta.append(x)

                Xaxis = np.arange(len(name))
                plt.barh(Xaxis, self.beta[:, 0])
                plt.yticks(Xaxis, name)
                plt.xlabel('Beta_1(R_m - R_f)')
                plt.title('Assets')
                plt.show()

                Xaxis = np.arange(len(name))
                plt.barh(Xaxis, self.beta[:, 1])
                plt.yticks(Xaxis, name)
                plt.xlabel('Beta_2(SML)')
                plt.title('Assets')
                plt.show()

                Xaxis = np.arange(len(name))
                plt.barh(Xaxis, self.beta[:, 2])
                plt.yticks(Xaxis, name)
                plt.xlabel('Beta_2(HML)')
                plt.title('Assets')
                plt.show()

                Xaxis = np.arange(len(name))
                plt.barh(Xaxis, self.beta[:, 3])
                plt.yticks(Xaxis, name)
                plt.xlabel('Beta_2(Momentum)')
                plt.title('Assets')
                plt.show()


            if i == 4:
                x = PrettyTable(["Asset", 'Beta_0(intercept)', 'Beta_1(R_m - R_f)', 'Beta_2(SML)', 'Beta_3(HML)', "Beta_4(China_us_exchange)"])
                x.align["Asset"] = "l"
                x.padding_width = 1
                self.FF(self.models[i])
                for j in range(len(name)):
                    x.add_row([name[j]] + [str(self.alpha[j])] + [str(self.beta[j][0])] + [str(self.beta[j][1])] +
                    [str(self.beta[j][2])] + [str(self.beta[j][3])])
                table_beta.append(x)

                Xaxis = np.arange(len(name))
                plt.barh(Xaxis, self.beta[:, 0])
                plt.yticks(Xaxis, name)
                plt.xlabel('Beta_1(R_m - R_f)')
                plt.title('Assets')
                plt.show()

                Xaxis = np.arange(len(name))
                plt.barh(Xaxis, self.beta[:, 1])
                plt.yticks(Xaxis, name)
                plt.xlabel('Beta_2(SML)')
                plt.title('Assets')
                plt.show()

                Xaxis = np.arange(len(name))
                plt.barh(Xaxis, self.beta[:, 2])
                plt.yticks(Xaxis, name)
                plt.xlabel('Beta_2(HML)')
                plt.title('Assets')
                plt.show()

                Xaxis = np.arange(len(name))
                plt.barh(Xaxis, self.beta[:, 3])
                plt.yticks(Xaxis, name)
                plt.xlabel('Beta_2(China_US_Exchange)')
                plt.title('Assets')
                plt.show()

            if i == 5:
                x = PrettyTable(["Asset", 'Beta_0(intercept)', 'Beta_1(R_m - R_f)', 'Beta_2(SML)', 'Beta_3(HML)', 'Beta_4(Momentum)', "Beta_5(China_us_exchange)"])
                x.align["Asset"] = "l"
                x.padding_width = 1
                self.FF(self.models[i])
                for j in range(len(name)):
                    x.add_row([name[j]] + [str(self.alpha[j])] + [str(self.beta[j][0])] + [str(self.beta[j][1])] + [str(self.beta[j][2])] +
                              [str(self.beta[j][3])] + [str(self.beta[j][4])])
                table_beta.append(x)

                Xaxis = np.arange(len(name))
                plt.barh(Xaxis, self.beta[:, 0])
                plt.yticks(Xaxis, name)
                plt.xlabel('Beta_1(R_m - R_f)')
                plt.title('Assets')
                plt.show()

                Xaxis = np.arange(len(name))
                plt.barh(Xaxis, self.beta[:, 1])
                plt.yticks(Xaxis, name)
                plt.xlabel('Beta_2(SML)')
                plt.title('Assets')
                plt.show()

                Xaxis = np.arange(len(name))
                plt.barh(Xaxis, self.beta[:, 2])
                plt.yticks(Xaxis, name)
                plt.xlabel('Beta_2(HML)')
                plt.title('Assets')
                plt.show()

                Xaxis = np.arange(len(name))
                plt.barh(Xaxis, self.beta[:, 3])
                plt.yticks(Xaxis, name)
                plt.xlabel('Momentum')
                plt.title('Assets')
                plt.show()

                Xaxis = np.arange(len(name))
                plt.barh(Xaxis, self.beta[:, 4])
                plt.yticks(Xaxis, name)
                plt.xlabel('Beta_2(China_US_Exchange)')
                plt.title('Assets')
                plt.show()


    def ex_return_table(self):
        name = ["AAL", "AAPL", "BABA", "BIDU", "BURL", "D", "DAL",
                "FB", "FDX", "FRT", "GNC", "GOOG", "GPRO", "L",
                "NKE", "O", "S", "SPWR", "T", "UPS"]
        table_return = PrettyTable(["Asset"] + self.models + ["Historical return"])
        for j in range(len(name)):
            temp = []
            for i in range(len(self.models)):
                if i <= 2:
                    self.capm_traditional(self.models[i])
                    temp.append(np.round(self.expected_return[j], 4))
                else:
                    self.FF(self.models[i])
                    temp.append(np.round(self.expected_return[j], 4))
            table_return.add_row([name[j]] + temp + [np.round(np.mean(self.train[j]), 4)])
        print(table_return)

        yaxis = [np.mean(item) for item in self.train[0:-1]]
        Xaxis = np.arange(len(name))
        plt.barh(Xaxis, yaxis)
        plt.yticks(Xaxis, name)
        plt.xlabel('Expected return by using mean value')
        plt.title('Assets')
        plt.show()


    def return_variance_table(self):
        name = ["AAL", "AAPL", "BABA", "BIDU", "BURL", "D", "DAL",
                "FB", "FDX", "FRT", "GNC", "GOOG", "GPRO", "L",
                "NKE", "O", "S", "SPWR", "T", "UPS"]
        table_return = PrettyTable(["Asset"] + self.models + ["Historical return"])
        for j in range(len(name)):
            variance_list = []
            temp = []
            for i in range(len(self.models)):
                if i <= 2:
                    self.capm_traditional(self.models[i])
                    temp.append(np.round(self.return_variance[j], 4))
                    variance_list.append(self.return_variance)
                else:
                    self.FF(self.models[i])
                    temp.append(np.round(self.return_variance[j], 4))
                    variance_list.append(self.return_variance)
            table_return.add_row([name[j]] + temp + [np.round(np.var(self.train[j]) / 210, 4)])
        print(table_return)
        variance_list = np.asarray(variance_list)
        difference_list = []
        for i in range(len(variance_list)):
            for j in range(i+1, len(variance_list)):
                difference_list.append(variance_list[i] - variance_list[j])

        name = ["AAL", "AAPL", "BABA", "BIDU", "BURL", "D", "DAL",
                "FB", "FDX", "FRT", "GNC", "GOOG", "GPRO", "L",
                "NKE", "O", "S", "SPWR", "T", "UPS"]
        Xaxis = np.arange(len(difference_list[i]))
        fig = plt.figure()
        ax = fig.add_subplot(111)

        for i in range(len(difference_list)):
            plt.plot(abs(difference_list[i]), Xaxis)
            ax.xaxis.tick_bottom()
            #ax.yaxis.tick_right()
            #print(difference_list[i])
        plt.yticks(Xaxis, name)
        plt.xlabel('Absolute difference value')
        plt.title('Assets')
        ax.set_xlim(0, 0.006)
        plt.show()







    def compare_cov_matrix(self):
        name = ["AAL", "AAPL", "BABA", "BIDU", "BURL", "D", "DAL",
                "FB", "FDX", "FRT", "GNC", "GOOG", "GPRO", "L",
                "NKE", "O", "S", "SPWR", "T", "UPS"]
        table_cov = PrettyTable(["Values"] + self.models + ["Historical return"])
        for i in range(len(self.models) + 1):
            if i <= len(self.models) - 1:
                rows = [self.models[i]]
            else:
                rows = ['Historical return']

            if i <= 2:
                self.capm_traditional(self.models[i])
                cov_1 = self.cov_matrix
                for j in range(len(self.models) + 1):
                    if j <= 2:
                        self.capm_traditional(self.models[j])
                        cov_2 = self.cov_matrix
                        rows.append(round(np.linalg.norm(cov_1 - cov_2, ord=2), 4))
                    elif j <= len(self.models) - 1:
                        self.FF(self.models[j])
                        cov_2 = self.cov_matrix
                        rows.append(round(np.linalg.norm(cov_1 - cov_2, ord=2),4))
                    else:
                        rows.append(round(np.linalg.norm(cov_1 - np.cov(self.train[0:-1]) /210, ord=2), 4))

            elif i <= len(self.models) - 1:
                self.FF(self.models[i])
                cov_1 = self.cov_matrix
                for j in range(len(self.models) + 1):
                    if j <= 2:
                        self.capm_traditional(self.models[j])
                        cov_2 = self.cov_matrix
                        rows.append(round(np.linalg.norm(cov_1 - cov_2, ord=2), 4))
                    elif j <= len(self.models) - 1:
                        self.FF(self.models[j])
                        cov_2 = self.cov_matrix
                        rows.append(round(np.linalg.norm(cov_1 - cov_2, ord=2), 4))
                    else:
                        rows.append(round(np.linalg.norm(cov_1 - np.cov(self.train[0:-1]) /210, ord=2), 4))
            else:
                cov_1 = np.cov(self.train[0:-1]) /210
                for j in range(len(self.models) + 1):
                    if j <= 2:
                        self.capm_traditional(self.models[j])
                        cov_2 = self.cov_matrix
                        rows.append(round(np.linalg.norm(cov_1 - cov_2, ord=2), 4))
                    elif j <= len(self.models) - 1:
                        self.FF(self.models[j])
                        cov_2 = self.cov_matrix
                        rows.append(round(np.linalg.norm(cov_1 - cov_2, ord=2), 4))
                    else:
                        rows.append(round(np.linalg.norm(cov_1 - np.cov(self.train[0:-1]) /210, ord=2), 4))
            table_cov.add_row(rows)
        print(table_cov)







## For 20 assets
# risk_free rate use 1 year date with initial date at 03/01/2016
solution = solver(0.0068, "AAL.csv", "AAPL.csv", "BABA.csv", "BIDU.csv", "BURL.csv", "D.csv", "DAL.csv",
                         "FB.csv", "FDX.csv", "FRT.csv", "GNC.csv", "GOOG.csv", "GPRO.csv", "L.csv",
                         "NKE.csv",  "O.csv", "S.csv", "SPWR.csv", "T.csv", "UPS.csv", "^GSPC.csv")
#solution = solver(0.0068, "AAL.csv", "AAPL.csv", "BABA.csv", "BIDU.csv", "BURL.csv", "D.csv", "DAL.csv",
#                         "FB.csv", "FDX.csv", "FRT.csv", "GNC.csv", "GPRO.csv", "L.csv",
#                         "NKE.csv",  "O.csv", "S.csv", "SPWR.csv", "T.csv", "UPS.csv", "^GSPC.csv")
solution.get_exchange("China_us_exchange2.csv")
solution.get_FF_factors("FF_Factors.CSV")
solution.get_momentume("Momentum_Factor.csv")
solution.get_price_list(3)
solution.cal_Sigma_F()

### Construct six models
#solution.construct_models()

### Estimates of expected return and variance
#solution.ex_return_table()
#solution.return_variance_table()
#solution.compare_cov_matrix()


#solution.capm_traditional("CAPM_F_1")
#print(solution.expected_return)
#print(solution.beta[0][0])
#print(len(solution.alpha))
#solution.construct_models()
#solution.compare_cov_matrix()




### Construct efficient frontier by CAMP plus F_1(Momentum) and F_2(exchang)
# ef_CAPM_F_1_F_2 = efficientFrontier()
# solution.capm_traditional("CAPM_F_1")
#
# ef_CAPM_F_1_F_2.set_expected_return(solution.expected_return)
# ef_CAPM_F_1_F_2.set_parameters(solution.cov_matrix)
# ef_CAPM_F_1_F_2.optimalWeights()
# ef_CAPM_F_1_F_2.get_tangencyPortfolio(0.002)
# weight = ef_CAPM_F_1_F_2.omega_T
#
# # name = ["AAL", "AAPL", "BABA", "BIDU", "BURL", "D", "DAL",
# #                 "FB", "FDX", "FRT", "GNC", "GOOG", "GPRO", "L",
# #                 "NKE", "O", "S", "SPWR", "T", "UPS"]
# # Yaxis = [item[0] for item in weight]
# # Xaxis = np.arange(len(name))
# # plt.barh(Xaxis, Yaxis)
# # plt.yticks(Xaxis, name)
# # plt.xlabel('Weight (CAPM Model)')
# # plt.title('Assets')
# # plt.show()
#
#
# start_price = [item[0] for item in solution.price_test]
# end_price = [item[-1] for item in solution.price_test]
# difference = np.array(end_price) - np.array(start_price)
# #print(difference)
# # #print(len(end_price))
# number = []
# for i in range(len(weight)):
#     MK_price = start_price[i]
#     number.append(int(500000 * weight[i] / MK_price))
# #print(number)
# final_value_capm = np.sum(np.asarray(end_price) * np.asarray(number))
# #print(final_value_capm)
# final_return_capm = (final_value_capm - 500000 )/500000 + 0.002
# predict_return_capm = ef_CAPM_F_1_F_2.predic
# #print(final_return_capm)
# #
# #
# daily_port_value_capm = [500000]
# for i in range(len(solution.price_test[0])):
#     price = [item[i] for item in solution.price_test]
#     daily_port_value_capm.append(np.sum(np.asarray(price) * np.asarray(number)))
# daily_port_return_capm = []
# for i in range(len(daily_port_value_capm) - 1):
#     daily_port_return_capm.append((daily_port_value_capm[i + 1] - daily_port_value_capm[i])/daily_port_value_capm[i])
# var_capm = np.var(daily_port_return_capm)
# #print(var_capm)
# #
#
# # ## Information ratio
# market_return = solution.test[-1]/210
# RI_capm = (np.mean(daily_port_return_capm) - np.mean(market_return))/np.std(daily_port_return_capm - market_return)
# print(RI_capm)
#
# ## Sharpe Measure
# RS_capm = (np.mean(daily_port_return_capm) - 0.002/210)/np.std(daily_port_return_capm)
# print(RS_capm)
#
# ## Treynor Measure
# beta_p = np.cov([daily_port_return_capm, market_return])[0][1] / np.var(market_return)
# RT_capm = (np.mean(daily_port_return_capm) - 0.002/210)/beta_p
# print(RT_capm)
#
# ##Sortino Ratio
# y_i = []
# for i in daily_port_return_capm:
#     if i >= 0.002/210:
#         y_i.append(0.002/210)
#     else:
#         y_i.append(i)
# std_y =(1/len(y_i))*sum((np.asarray(y_i) - 0.002/210)**2)
# std_y = np.sqrt(std_y)
# SR_capm = (np.mean(daily_port_return_capm) - 0.002/210)/std_y
# print(SR_capm)


#
# ## 20 var
# var_20_capm = solution.return_variance
# cov_capm = solution.cov_matrix

## 20 assets sharpe ratio


# print('----------')
#
# ### Construct efficient frontier by FF plus F_1(Momentum) and F_2(exchang)
# ef_FF_F_1_F_2 = efficientFrontier()
# solution.FF("FF_F_1_F_2")
# ef_FF_F_1_F_2.set_expected_return(solution.expected_return)
# ef_FF_F_1_F_2.set_parameters(solution.cov_matrix)
# ef_FF_F_1_F_2.optimalWeights()
# ef_FF_F_1_F_2.get_tangencyPortfolio(0.002)
# weight = ef_FF_F_1_F_2.omega_T
# start_price = [item[0] for item in solution.price_test]
# end_price = [item[-1] for item in solution.price_test]
# difference = np.array(end_price) - np.array(start_price)
# #print(difference)
# number = []
# for i in range(len(weight)):
#     MK_price = start_price[i]
#     number.append(int(500000 * weight[i] / MK_price))
# #print(number)
# final_value_ff = np.sum(np.asarray(end_price) * np.asarray(number))
# #print(final_value_ff)
# final_return_ff = (final_value_ff - 500000 )/500000 + 0.002
# #print(final_return_ff)
# predict_return_ff = ef_FF_F_1_F_2.predic
# #
# daily_port_value_ff = [500000]
# for i in range(len(solution.price_test[0])):
#     price = [item[i] for item in solution.price_test]
#     daily_port_value_ff.append(np.sum(np.asarray(price) * np.asarray(number)))
# daily_port_return_ff = []
# for i in range(len(daily_port_value_ff) - 1):
#     daily_port_return_ff.append((daily_port_value_ff[i + 1] - daily_port_value_ff[i])/daily_port_value_ff[i])
# #print(daily_port_return_capm)
# var_ff = np.var(daily_port_return_ff)
# #print(var_ff)
# #
# # ## Information ratio
# market_return = solution.test[-1]/210
# RI_ff = (np.mean(daily_port_return_ff) - np.mean(market_return))/np.std(daily_port_return_ff - market_return)
# print(RI_ff)
#
# ## Sharpe Measure
# RS_ff = (np.mean(daily_port_return_ff) - 0.002/210)/np.std(daily_port_return_ff)
# print(RS_ff)
#
# ## Treynor Measure
# beta_p = np.cov([daily_port_return_ff, market_return])[0][1] / np.var(market_return)
# RT_ff = (np.mean(daily_port_return_ff) - 0.002/210)/beta_p
# print(RT_ff)
#
# ##Sortino Ratio
# y_i = []
# for i in daily_port_return_ff:
#     if i >= 0.002/210:
#         y_i.append(0.002/210)
#     else:
#         y_i.append(i)
# std_y =(1/len(y_i))*sum((np.asarray(y_i) - 0.002/210)**2)
# std_y = np.sqrt(std_y)
# SR_ff = (np.mean(daily_port_return_ff) - 0.002/210)/std_y
# print(SR_ff)


#
# ## 20 var
# var_20_ff = solution.return_variance
# cov_ff = solution.cov_matrix
# print("----------")




### Construct efficient frontier by historical data
# ef_historical = efficientFrontier()
# expected_return = [np.mean(item) for item in solution.train[0:-1]]
# ef_historical.set_expected_return(expected_return)
# ef_historical.set_parameters(np.cov(solution.train[0:-1])/210)
# ef_historical.optimalWeights()
# ef_historical.get_tangencyPortfolio(0.002)
# weight = ef_historical.omega_T
# start_price = [item[0] for item in solution.price_test]
# end_price = [item[-1] for item in solution.price_test]
# difference = np.array(end_price) - np.array(start_price)
# #print(difference)
# number = []
# for i in range(len(weight)):
#     MK_price = start_price[i]
#     number.append(int(500000 * weight[i] / MK_price))
# #print(number)
# final_value_historical = np.sum(np.asarray(end_price) * np.asarray(number))
# #print(final_value_historical)
# final_return_historical = (final_value_historical - 500000)/500000 + 0.02
# #print(final_return_historical)
# predict_return_historical = ef_historical.predic
# #
# daily_port_value_historical = [500000]
# for i in range(len(solution.price_test[0])):
#     price = [item[i] for item in solution.price_test]
#     daily_port_value_historical.append(np.sum(np.asarray(price) * np.asarray(number)))
# daily_port_return_historical = []
# for i in range(len(daily_port_value_ff) - 1):
#     daily_port_return_historical.append((daily_port_value_historical[i + 1] - daily_port_value_historical[i])/daily_port_value_historical[i])
# #print(daily_port_return_historical)
# var_historical = np.var(daily_port_return_historical)
# #print(var_historical)
# #
#
# ## Information ratio
# market_return = solution.test[-1]/210
# RI_historical = (np.mean(daily_port_return_historical) - np.mean(market_return))/np.std(daily_port_return_historical - market_return)
# print(RI_historical)
#
# ## Sharpe Measure
# RS_historical = (np.mean(daily_port_return_historical) - 0.002/210)/np.std(daily_port_return_historical)
# print(RS_historical)
#
# ## Treynor Measure
# beta_p = np.cov([daily_port_return_historical, market_return])[0][1] / np.var(market_return)
# RT_historical = (np.mean(daily_port_return_historical) - 0.002/210)/beta_p
# print(RT_historical)

##Sortino Ratio
# y_i = []
# for i in daily_port_return_historical:
#     if i >= 0.002/210:
#         y_i.append(0.002/210)
#     else:
#         y_i.append(i)
# std_y =(1/len(y_i))*sum((np.asarray(y_i) - 0.002/210)**2)
# std_y = np.sqrt(std_y)
# SR_historical = (np.mean(daily_port_return_historical) - 0.002/210)/std_y
# print(SR_historical)

#
# ## 20 var, cov
# var_20_historical = []
# for i in range(len(solution.train[0:-1])):
#     var_20_historical.append(np.var(solution.train[0:-1][i]) /210)
# cov_historical = np.cov(solution.train[0:-1]) /210
#print('--------')




## Performance table
# table_per = PrettyTable(["Models", 'CAPM_F_1', 'FF_F_1_F_2', "Historical"])
# table_per.align["Models"] = "l"
# table_per.padding_width = 1
# table_per.add_row(['Sharpe Measure', round(RS_capm, 4) ,
#                            round(RS_ff, 4), round(RS_historical, 4)])
# table_per.add_row(['Treynor Measure',round(RT_capm, 4) ,
#                            round(RT_ff, 4), round(RT_historical, 4)])
# table_per.add_row(['Information Ratio', round(RI_capm, 4) ,
#                            round(RI_ff, 4), round(RI_historical, 4)])
# table_per.add_row(['Sortino Ratio',round(SR_capm, 4) ,
#                            round(SR_ff, 4), round(SR_historical, 4)])
# print(table_per)
#
# ## Performance plot
# names = ['CAPM','FF', 'Historical']
# RS = [RS_capm , RS_ff, RS_historical]
# RT = [RT_capm , RT_ff, RT_historical]
# RI = [RI_capm , RI_ff, RI_historical]
# SR = [SR_capm , SR_ff, SR_historical]
# Xaxis = np.arange(len(names))
# plt.bar(Xaxis, RS)
# plt.xticks(Xaxis, names)
# plt.ylabel('Sharpe Measure')
# plt.title('Models')
# plt.show()
# plt.bar(Xaxis, RT)
# plt.xticks(Xaxis, names)
# plt.ylabel('Treynor Measure')
# plt.title('Models')
# plt.show()
# plt.bar(Xaxis, RI)
# plt.xticks(Xaxis, names)
# plt.ylabel('Information Ratio')
# plt.title('Models')
# plt.show()
# plt.bar(Xaxis, SR)
# plt.xticks(Xaxis, names)
# plt.ylabel('Sortino Ratio')
# plt.title('Models')
# plt.show()




## final return table
# port_return_table = PrettyTable(["Models", 'CAPM_F_1_F_2', 'FF_F_1_F_2', "Historical"])
# port_return_table.align["Models"] = "l"
# port_return_table.padding_width = 1
# port_return_table.add_row(['Actual return', round(final_return_capm, 4),
#                            round(final_return_ff, 4), round(final_return_historical, 4)])
# port_return_table.add_row(['Estimate return', round(predict_return_capm, 4),
#                            round(predict_return_ff, 4), round(predict_return_historical, 4)])
# print(port_return_table)


## portfolio variance hist
# names = ['CAPM','FF', 'Historical']
# var = [var_capm, var_ff, var_historical]
# Xaxis = np.arange(len(var))
# plt.bar(Xaxis, var)
# plt.xticks(Xaxis, names)
# plt.ylabel('variance')
# plt.title('Models')
# plt.show()
#
# print(np.cov([var_capm, var_ff, var_historical]))



## plot portfolio return
# print(solution.price_test)
# plt.plot(np.arange(len(solution.price_test[0])), daily_port_return_capm)
# plt.show()
# plt.plot(np.arange(len(solution.price_test[0])), daily_port_return_ff)
# plt.show()
# plt.plot(np.arange(len(solution.price_test[0])), daily_port_return_historical)
# plt.show()


## compare expected parameter and actual parameter

# actual_return = solution.test[0:-1]
# #print(np.var(actual_return[0]))
# actual_var = []
# for i in range(len(actual_return)):
#     actual_var.append(np.var(actual_return[i]) / 210)
# actual_cov = np.cov(actual_return) / 210
# #print(actual_var)
#
# ## Variance
# name = ["AAL", "AAPL", "BABA", "BIDU", "BURL", "D", "DAL",
#                 "FB", "FDX", "FRT", "GNC", "GOOG", "GPRO", "L",
#                 "NKE", "O", "S", "SPWR", "T", "UPS"]
# var_table = PrettyTable(["Values", 'CAPM return variance', 'FF return variance', "Historical return variance", 'Actual return variance'])
# var_table.align["Values"] = "l"
# var_table.padding_width = 1
# for i in range(len(name)):
#     var_table.add_row([name[i], round(var_20_capm[i], 4), round(var_20_ff[i], 4), round(var_20_historical[i], 4), round(actual_var[i],  4) ])
# print(var_table)
#
# plt.barh(np.arange(len(name)), var_20_ff)
# plt.barh(np.arange(len(name)), actual_var)
# plt.legend(["Model", " Actual"])
# plt.yticks(np.arange(len(name)), name)
# plt.xlabel('Return Variance')
# plt.title('Assets')
# plt.show()




## expected return
# name = ["AAL", "AAPL", "BABA", "BIDU", "BURL", "D", "DAL",
#                 "FB", "FDX", "FRT", "GNC", "GOOG", "GPRO", "L",
#                 "NKE", "O", "S", "SPWR", "T", "UPS"]
# compare_return_table = PrettyTable(["Assets", 'Model return', 'Actual return'])
# compare_return_table.align["Assets"] = "l"
# compare_return_table.padding_width = 1
# for i in range(len(name)):
#     compare_return_table.add_row([name[i], round(solution.expected_return[i], 4), round(np.mean(actual_return[i]), 4)])
# print(compare_return_table)
#
# difference = []
# for i in range(len(actual_return)):
#     difference.append(solution.expected_return[i] - np.mean(actual_return[i]))
# plt.barh(np.arange(len(name)), difference)
# plt.yticks(np.arange(len(name)), name)
# plt.xlabel('Difference')
# plt.title('Assets')
# plt.show()


## Covariance
# name = ["AAL", "AAPL", "BABA", "BIDU", "BURL", "D", "DAL",
#                 "FB", "FDX", "FRT", "GNC", "GOOG", "GPRO", "L",
#                 "NKE", "O", "S", "SPWR", "T", "UPS"]
# cov_table = PrettyTable(["Models", 'CAPM ', 'FF', "Historical return variance"])
# cov_table.align["Models"] = "l"
# cov_table.padding_width = 1
# cov_table.add_row(["Actual", round(np.linalg.norm(cov_capm - actual_cov, ord = 2), 4),
#                    round(np.linalg.norm(cov_ff - actual_cov, ord=2), 4),
#                    round(np.linalg.norm(cov_historical - actual_cov, ord = 2), 4)])
# print(cov_table)



## Quantile-Based back-testing
#ranking
name = ["AAL", "AAPL", "BABA", "BIDU", "BURL", "D", "DAL",
                 "FB", "FDX", "FRT", "GNC", "GOOG", "GPRO", "L",
                 "NKE", "O", "S", "SPWR", "T", "UPS"]
## CAPM
solution.capm_traditional("CAPM_F_1")
expected_return = np.asarray(solution.expected_return)
ratio = (expected_return - 0.002)/np.sqrt(solution.return_variance)
#print(ratio)

ratio_temp = []
name_temp = []
copy = np.copy(ratio)
copy2 = np.copy(name)
for i in range(len(name)):
    temp = min(copy)
    index = np.nonzero(copy == temp)[0][0]
    ratio_temp.append(temp)
    name_temp.append(copy2[index])
    copy = np.delete(copy, index)
    copy2 = np.delete(copy2, index)

plt.barh(np.arange(len(name_temp)), ratio_temp)
plt.yticks(np.arange(len(name)), name_temp)
plt.xlabel('Sharpe ratio')
plt.title('Assets')
plt.show()
#print(name_temp)

## new portfolio
top  = [13, 16, 19]
bottom = [10, 14, 17]
sum_asset = top + bottom
cov_capm = solution.cov_matrix
top_expected_return = [expected_return[item] for item in range(len(expected_return)) if item in top]
bottom_expected_return = [expected_return[item] for item in range(len(expected_return)) if item in bottom]

start_price = [item[0] for item in solution.price_test]
end_price = [item[-1] for item in solution.price_test]
difference = np.array(end_price) - np.array(start_price)
difference_top = [difference[item] for item in range(len(difference)) if item in top]
difference_bottom = [difference[item] for item in range(len(difference)) if item in bottom]



random = np.linspace(1/3, 1,10000)

list = []
for i in range(len(random)):
    if i == 100:
        weight_t = random[i]
        weight_b = 1/3 - weight_t
        weight = [weight_t, weight_t, weight_t, weight_b, weight_b, weight_b]
        list.append(sum(np.asarray([weight_t, weight_t, weight_t]) * np.asarray(difference_top)) +
                    sum(np.asarray([weight_b, weight_b, weight_b]) * np.asarray(difference_bottom)))
        number = []
        MK_price_top = [start_price[item] for item in range(len(start_price)) if item in top]
        MK_price_bottom = [start_price[item] for item in range(len(start_price)) if item in bottom]
        MK_price = MK_price_top + MK_price_bottom
        number = 500000 * np.asarray(weight) / np.asarray(MK_price)
        daily_port_value_capm = [500000]

        price_top = [solution.price_test[item] for item in range(len(solution.price_test)) if item in top]
        price_bottom = [solution.price_test[item] for item in range(len(solution.price_test)) if item in bottom]
        price = price_top + price_bottom
        daily_port_value_capm = np.dot(np.asarray(price).T , np.asarray(number))

        daily_port_return_capm = []
        for j in range(len(daily_port_value_capm) - 1):
            daily_port_return_capm.append(
                (daily_port_value_capm[j + 1] - daily_port_value_capm[j]) / daily_port_value_capm[j])
        daily_port_return_capm = np.asarray(daily_port_return_capm)
        daily_port_value_capm = np.asarray(daily_port_value_capm)
        #print(daily_port_return_capm)
        SR = (np.mean(daily_port_return_capm) - 0.002) / np.std(daily_port_return_capm)
        plt.plot(np.arange(len(daily_port_return_capm)), daily_port_return_capm)
        plt.show()


#print(list)
# print(difference_top)
# print(difference_bottom)
# print(difference)
# plt.plot(random, list)
# plt.show()





# list = []
# for i in range(len(random)):
#     weight = random[i]
#     sigma = 0
#     for j in range(len(top)):
#         for k in range(j,len(bottom)):
#             sigma = sigma + cov_capm[j][k]
#     Sigma = (weight*(1-weight)*sigma)
#     list.append((weight*sum(top_expected_return) + (1-weight)*sum(bottom_expected_return) - 0.002)/
#                 Sigma)
# plt.plot(np.arange(len(random)), list)
# plt.show()
















