import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from prettytable import PrettyTable

class capm:
    def __init__(self, *files):
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
        self.name = []
        self.portfolio_beta = []


    def get_price_list(self, years):
        index = years/3
        for file in self.files:
            csv = pd.read_csv(file)
            csv.drop(['Date', 'Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1, inplace=True)
            list = []
            for item in csv.as_matrix():
                list.append(item[0])
            self.price_list.append(list[-int(index * len(list)):len(list)])

        for i in self.price_list:
            returns = []
            for j in range(len(i) - 1):
                returns.append((i[j + 1] - i[j]) / i[j])
            self.returns.append(np.log(1 + np.asarray(returns)))


    def annulize(self, array):
        array = np.asarray(array)
        return array*252

    def get_expected_return(self):
        for item in self.returns[0:-1]:
            self.expected_return.append(self.annulize(np.mean(item)))
        self.expected_return = np.asarray(self.expected_return)

    def cal_weights(self, risk_free):
        self.Omega = np.cov(self.returns[0:-1])
        self.Omega = self.annulize(self.Omega)
        for item in self.expected_return:
            self.mu_vector.append([item])
        self.mu_f = risk_free
        ones = np.ones([len(self.Omega), 1])
        self.omega_bar = np.dot(np.linalg.inv(self.Omega), self.mu_vector - self.mu_f * ones)
        self.omega_T = self.omega_bar / np.dot(np.transpose(ones), self.omega_bar)


    def cal_parameters_historical(self, risk_free):
        self.beta = []
        for i in range(len(self.returns[0:-1])):
            assets_return = self.returns[i:-1]
            assets_return_star = np.asarray(assets_return) - risk_free
            market_return = self.returns[-1]
            market_return_star = np.asarray(market_return) - risk_free
            cov_matrix = np.cov(self.returns)
            beta = cov_matrix[-1][i] / cov_matrix[-1][-1]
            alpha = np.mean(assets_return_star) - beta * np.mean(market_return_star)
            self.beta.append(round(beta, 4))
            self.alpha.append(alpha)
        self.portfolio_beta = np.dot(self.beta, self.omega_T)

        name = ["AAL", "AAPL", "BABA", "BIDU", "BURL", "D", "DAL",
                "FB", "FDX", "FRT", "GNC", "GOOG", "GPRO", "L",
                "NKE", "O", "S", "SPWR", "T", "UPS"]
        ## Table
        beta_temp = []
        name_temp = []
        copy = np.copy(self.beta)
        copy2 = np.copy(name)
        for i in range(len(name)):
            temp = min(copy)
            index = np.nonzero(copy == temp)[0][0]
            beta_temp.append(temp)
            name_temp.append(copy2[index])
            copy = np.delete(copy, index)
            copy2 = np.delete(copy2, index)
        table = PrettyTable(["Assets", 'Beta'])
        table.align["Models"] = "l"
        table.padding_width = 1
        for i in range(len(name_temp)):
            table.add_row([name_temp[i], beta_temp[i]])
        #print(table)

        ## Plot
        plt.barh(np.arange(len(name)), beta_temp)
        plt.yticks(np.arange(len(name)), name_temp)
        for i, v in enumerate(beta_temp):
            plt.text(v + 0, i + .25, str(v), color='blue', fontweight='bold')
        plt.gca().invert_yaxis()
        plt.plot(np.ones(len(name)), np.arange(len(name)), color = "red")
        plt.xlabel('Beta')
        plt.title('Assets')

        plt.show()
        #print(self.beta)
        #print(round(portfolio_beta[0], 4))

    def cal_paremeters_shrinkage(self, risk_free, alpha_i):
        self.beta = []
        cov_matrix = np.cov(self.returns)
        for i in range(len(self.returns[0:-1])):
            assets_return = self.returns[i:-1]
            assets_return_star = np.asarray(assets_return) - risk_free
            market_return = self.returns[-1]
            market_return_star = np.asarray(market_return) - risk_free
            beta = cov_matrix[-1][i] / cov_matrix[-1][-1]
            self.beta.append(round(beta, 4))
        self.beta = (1 - alpha_i) * np.mean(self.beta) + alpha_i * np.asarray(self.beta)
        self.beta = np.round(self.beta, 4)
        self.portfolio_beta = np.dot(self.beta, self.omega_T)
        name = ["AAL", "AAPL", "BABA", "BIDU", "BURL", "D", "DAL",
                "FB", "FDX", "FRT", "GNC", "GOOG", "GPRO", "L",
                "NKE", "O", "S", "SPWR", "T", "UPS"]
        ## Table
        beta_temp = []
        name_temp = []
        copy = np.copy(self.beta)
        copy2 = np.copy(name)
        for i in range(len(name)):
            temp = min(copy)
            index = np.nonzero(copy == temp)[0][0]
            beta_temp.append(temp)
            name_temp.append(copy2[index])
            copy = np.delete(copy, index)
            copy2 = np.delete(copy2, index)
        table = PrettyTable(["Assets", 'Beta'])
        table.align["Models"] = "l"
        table.padding_width = 1
        for i in range(len(name_temp)):
            table.add_row([name_temp[i], beta_temp[i]])
        #print(table)

        ## Plot
        plt.barh(np.arange(len(name)), beta_temp)
        plt.yticks(np.arange(len(name)), name_temp)
        for i, v in enumerate(beta_temp):
            plt.text(v + 0, i + .25, str(v), color='blue', fontweight='bold')
        plt.gca().invert_yaxis()
        plt.plot(np.ones(len(name)), np.arange(len(name)), color="red")
        plt.xlabel('Beta')
        plt.title('Assets')

        plt.show()
        #print(cov_matrix[0])

    def cal_parameters_EWMA(self, risk_free, lbd):
        self.beta = []
        weight = []
        weight.append(1 - lbd)
        for i in range(len(self.returns[0]) - 1):
            prev = weight[-1]
            weight.append(prev * lbd)
        weight.reverse()

        covariance = []
        for i in range(len(self.files)):
            covarianceList = []
            for j in range(len(self.files)):
                returnX = np.asarray(self.returns[i])
                returnY = np.asarray(self.returns[j])
                covarianceList.append(np.sum(returnX * returnY * np.asarray(weight)))
            covariance.append(covarianceList)

        cov_matrix = covariance
        for i in range(len(self.returns[0:-1])):
            assets_return = self.returns[i:-1]
            assets_return_star = np.asarray(assets_return) - risk_free
            market_return = self.returns[-1]
            market_return_star = np.asarray(market_return) - risk_free
            beta = cov_matrix[-1][i] / cov_matrix[-1][-1]
            self.beta.append(round(beta,4))
        self.portfolio_beta = np.dot(self.beta, self.omega_T)

        name = ["AAL", "AAPL", "BABA", "BIDU", "BURL", "D", "DAL",
                "FB", "FDX", "FRT", "GNC", "GOOG", "GPRO", "L",
                "NKE", "O", "S", "SPWR", "T", "UPS"]
        ## Table
        beta_temp = []
        name_temp = []
        copy = np.copy(self.beta)
        copy2 = np.copy(name)
        for i in range(len(name)):
            temp = min(copy)
            index = np.nonzero(copy == temp)[0][0]
            beta_temp.append(temp)
            name_temp.append(copy2[index])
            copy = np.delete(copy, index)
            copy2 = np.delete(copy2, index)
        table = PrettyTable(["Assets", 'Beta'])
        table.align["Models"] = "l"
        table.padding_width = 1
        for i in range(len(name_temp)):
            table.add_row([name_temp[i], beta_temp[i]])
        #print(table)

        ## Plot
        plt.barh(np.arange(len(name)), beta_temp)
        plt.yticks(np.arange(len(name)), name_temp)
        for i, v in enumerate(beta_temp):
            plt.text(v + 0, i + .25, str(v), color='blue', fontweight='bold')
        plt.gca().invert_yaxis()
        plt.plot(np.ones(len(name)), np.arange(len(name)), color="red")
        plt.xlabel('Beta')
        plt.title('Assets')

        plt.show()

    def cal_parameters_LTS(self, risk_free):
        df = pd.DataFrame(self.returns[1])
        alpha_5 = df.quantile(0.05)
        alpha_95 = df.quantile(0.95)
        alpha_5 = alpha_5.values[0]
        alpha_95 = alpha_95.values[0]
        new_AAPL = np.copy(self.returns[1])
        for i in range(len(self.returns[1])):
            if new_AAPL[i] <= alpha_5:
                new_AAPL[i] = alpha_5
            if new_AAPL[i] >= alpha_95:
                new_AAPL[i] = alpha_95
        bins = np.linspace(-0.1, 0.1, 20)
        plt.hist(self.returns[1], bins)
        plt.show()
        plt.hist(new_AAPL, bins)
        plt.show()

    def mysort(self, name):
        name = ["AAL", "AAPL", "BABA", "BIDU", "BURL", "D", "DAL",
                "FB", "FDX", "FRT", "GNC", "GOOG", "GPRO", "L",
                "NKE", "O", "S", "SPWR", "T", "UPS"]
        ## Table
        beta_temp = []
        name_temp = []
        copy = np.copy(self.beta)
        copy2 = np.copy(name)
        for i in range(len(name)):
            temp = min(copy)
            index = np.nonzero(copy == temp)[0][0]
            beta_temp.append(temp)
            name_temp.append(copy2[index])
            copy = np.delete(copy, index)
            copy2 = np.delete(copy2, index)
        self.beta = beta_temp
        self.name = name_temp

## For 20 assets
solution = capm("AAL.csv", "AAPL.csv", "BABA.csv", "BIDU.csv", "BURL.csv", "D.csv", "DAL.csv",
                         "FB.csv", "FDX.csv", "FRT.csv", "GNC.csv", "GOOG.csv", "GPRO.csv", "L.csv",
                         "NKE.csv",  "O.csv", "S.csv", "SPWR.csv", "T.csv", "UPS.csv", "^GSPC.csv")
solution.get_price_list(1)
solution.get_expected_return()
solution.cal_weights(0.0168)
#solution.cal_parameters_historical(0.0168)
#solution.cal_parameters_LTS(0.0168)
#solution.cal_paremeters_shrinkage(0.0168, 2/3)
#solution.cal_parameters_EWMA(0.0168, 0.97)


name = ["AAL", "AAPL", "BABA", "BIDU", "BURL", "D", "DAL",
                "FB", "FDX", "FRT", "GNC", "GOOG", "GPRO", "L",
                "NKE", "O", "S", "SPWR", "T", "UPS"]
solution.cal_parameters_historical(0.0168)
#solution.mysort(name)
beta_temp4 = solution.beta
pb4 = np.round(solution.portfolio_beta, 4)
solution.cal_paremeters_shrinkage(0.0168, 2/3)
#solution.mysort(name)
beta_temp3 = solution.beta
pb3 = np.round(solution.portfolio_beta, 4)
solution.cal_parameters_EWMA(0.0168, 0.94)
#solution.mysort(name)
pb2 = np.round(solution.portfolio_beta, 4)
beta_temp = solution.beta
solution.cal_parameters_EWMA(0.0168, 0.97)
pb1 = np.round(solution.portfolio_beta, 4)
#solution.mysort(name)
beta_temp2 = solution.beta


# table = PrettyTable(["Values", 'EWMA(lambda = 0.94)', 'EWMA(lambda = 0.97)', "Shrinkage", "Traditional"])
# table.align["Values"] = "l"
# table.padding_width = 1
# for i in range(len(name)):
#     table.add_row([name[i], beta_temp[i], beta_temp2[i], beta_temp3[i], beta_temp4[i]])
# print(table)

table2 = PrettyTable(["Values", 'EWMA(lambda = 0.94)', 'EWMA(lambda = 0.97)', "Shrinkage", "Traditional"])
table2.align["Values"] = "l"
table2.padding_width = 1
table2.add_row( ['Portfolio Beta', pb1[0], pb2[0], pb3[0], pb4[0]])
print(table2)



























