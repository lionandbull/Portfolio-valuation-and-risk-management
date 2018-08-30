import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

class efficientFrontier:
    def __init__(self, *files):
        self.files = files
        self.price_list = []
        self.expected_return = []
        self.returns = []
        self.mu_vector = []
        self.Omega = []
        self.mu_min = 0
        self.sd_min = 0
        self.omega_bar = 0
        self.omega_T = 0
        self.sigma_T = 0
        self.mu_f = 0
        self.mu_T = 0
        self.g = 0
        self.h = 0
        self.num_stocks = []
        self.stock_prices = []
        self.names = []
        self.efficient_mu = []
        self.efficient_sigma = []
        self.unefficient_mu = []
        self.unefficient_sigma = []
        self.mu_p_star = []
        self.predic = 0

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


    def set_expected_return(self, expected_return):
        for item in expected_return:
            # self.expected_return.append(np.sqrt(365)*item)
            self.expected_return.append(item)
        self.expected_return = np.asarray(self.expected_return)


    def set_parameters(self, return_cov):
        self.Omega = return_cov
        #self.expected_return = np.array([0.07, 0.12, 0.09])
        #self.Omega = np.array([[0.2,0,0], [0, 0.3, 0], [0,0,0.25]])
        for item in self.expected_return:
            self.mu_vector.append([item])
        mu_vector = np.asarray(self.mu_vector)
        Omega = np.asarray(self.Omega)

        #mu_vector = np.asarray([[0.07], [0.12], [0.09]])
        #Omega = np.asarray([[0.2,0,0],[0,0.3,0],[0,0,0.25]])

        #mu_vector = np.asarray([[0.0427], [0.0015], [0.0285]])
        #Omega = np.asarray([[0.01,0.0018,0.0011],[0.0018,0.0109,0.0026],[0.0011,0.0026,0.0199]])

        if np.linalg.det(Omega) == 0:
            print("Omega is not reversible !!!")
            return
        ones = np.ones([len(Omega), 1])
        ## A = B, B =A, C = C
        A = self.threeMatrixDot(np.transpose(ones), np.linalg.inv(Omega), mu_vector)[0][0]
        B = self.threeMatrixDot(np.transpose(mu_vector), np.linalg.inv(Omega), mu_vector)[0][0]
        C = self.threeMatrixDot(np.transpose(ones), np.linalg.inv(Omega), ones)[0][0]
        D = np.dot(B, C) - np.dot(A, A)

        self.g = (B * np.dot(np.linalg.inv(Omega), ones) - A * np.dot(np.linalg.inv(Omega), mu_vector)) / D
        self.h = (C * np.dot(np.linalg.inv(Omega), mu_vector) - A * np.dot(np.linalg.inv(Omega), ones)) / D


        #test = (A*0.0089 - B)/(C*0.0089 - A)
        #test2 = (B - 2*0.0089*A +(0.0089**2*C))/((A-C*0.0089)**2)
        #print("--------")
        #print(test)
        #print(np.sqrt(test2))
        #return
        ###################### Test


        # gg = self.threeMatrixDot(np.transpose(self.g), Omega, self.g)[0][0]
        # hh = self.threeMatrixDot(np.transpose(self.h), Omega, self.h)[0][0]
        # gh = self.threeMatrixDot(np.transpose(self.g), Omega, self.h)[0][0]
        #
        # mu_min = - gh / hh
        # mu_min = mu_min
        # sd_min = np.sqrt(gg - (gh ** 2) / hh)
        # sd_min = sd_min
        #
        # mu_p_star = np.linspace(min(mu_vector)[0], max(mu_vector)[0], 50)
        # # mu_p_star = np.linspace(min(self.mu_vector)[0], 3, 500)
        #
        # sigma_p = []
        # for i in range(len(mu_p_star)):
        #     omega_star = self.g + mu_p_star[i] * self.h
        #     sigma_p.append(np.sqrt(self.threeMatrixDot(np.transpose(omega_star), Omega, omega_star))[0][0])
        #
        # above_min = [i for i in range(len(mu_p_star)) if mu_p_star[i] > mu_min]
        # efficient_mu = [mu_p_star[i] for i in above_min]
        # efficient_sigma = [sigma_p[i] for i in above_min]
        # below_min = [j for j in range(len(mu_p_star)) if mu_p_star[j] < mu_min]
        # unefficient_mu = [mu_p_star[j] for j in below_min]
        # unefficient_sigma = [sigma_p[j] for j in below_min]
        #
        # alpha = np.linspace(-2, 1.5, 100)
        # Z_mu = []
        # Z_sigma = []
        # for i in range(len(alpha)):
        #     temp_1 = alpha[i] * mu_min + (1-alpha[i]) * (mu_p_star[-1])
        #     Z_mu.append(temp_1)
        #     temp_2 = self.g + temp_1 * self.h
        #     Z_sigma.append(np.sqrt(self.threeMatrixDot(np.transpose(temp_2), Omega, temp_2))[0][0])
        #
        # print(Z_mu)
        # print(Z_sigma)
        #
        # above_min = [i for i in range(len(Z_mu)) if Z_mu[i] > mu_min]
        # efficient_mu = [Z_mu[i] for i in above_min]
        # efficient_sigma = [Z_sigma[i] for i in above_min]
        # below_min = [j for j in range(len(Z_mu)) if Z_mu[j] < mu_min]
        # unefficient_mu = [Z_mu[j] for j in below_min]
        # unefficient_sigma = [Z_sigma[j] for j in below_min]
        #
        # plt.plot(efficient_sigma, efficient_mu, '-', color="black", linewidth=2)
        # #plt.plot(Z_mu, Z_sigma, '-', color="black", linewidth=2)
        # plt.plot(unefficient_sigma, unefficient_mu, '--', color="black")
        # plt.scatter(sd_min, mu_min, s=300, color="red", marker='*')
        #
        # plt.ylabel('Expected portfolio returns')
        # plt.xlabel('Standard deviation of return')
        # plt.title('Assets')
        #
        #
        # mu_f = 0.001
        # ones = np.ones([len(Omega), 1])
        # self.omega_bar = np.dot(np.linalg.inv(Omega), mu_vector - mu_f * ones)
        # self.omega_T = self.omega_bar / np.dot(np.transpose(ones), self.omega_bar)
        # self.mu_T = np.dot(np.transpose(mu_vector), self.omega_T)[0][0]
        # self.sigma_T = np.sqrt(self.threeMatrixDot(np.transpose(self.omega_T),
        #                                            Omega,
        #                                            self.omega_T))
        # plt.plot([0, self.sigma_T], [mu_f, self.mu_T])
        # plt.show()
        #
        # return


        # if self.mu_T in mu_p_star:
        #     print("GOOD")
        # else:
        #     print("BAD")
        # return


    def threeMatrixDot(self, a, b, c):
        return np.dot(np.dot(a, b), c)

    def optimalPortfolio(self, mu_p):
        return self.g + mu_p * self.h

    def optimalWeights(self):
        mu_p_star = np.linspace(min(self.mu_vector)[0], max(self.mu_vector)[0], 50)
        #mu_p_star = np.linspace(min(self.mu_vector)[0], 3, 500)

        sigma_p = []
        for i in range(len(mu_p_star)):
            omega_star = self.g + mu_p_star[i] * self.h
            sigma_p.append(np.sqrt(self.threeMatrixDot(np.transpose(omega_star), self.Omega, omega_star))[0][0])

        gg = self.threeMatrixDot(np.transpose(self.g), self.Omega, self.g)[0][0]
        hh = self.threeMatrixDot(np.transpose(self.h), self.Omega, self.h)[0][0]
        gh = self.threeMatrixDot(np.transpose(self.g), self.Omega, self.h)[0][0]

        mu_min = - gh / hh
        mu_min = mu_min
        sd_min = np.sqrt(gg - (gh**2) / hh)
        sd_min = sd_min

        self.mu_min = mu_min
        self.sd_min = sd_min
       # print(self.mu_min, self.sd_min)

        above_min = [i for i in range(len(mu_p_star)) if mu_p_star[i] > mu_min]
        efficient_mu = [mu_p_star[i] for i in above_min ]
        efficient_sigma = [sigma_p[i] for i in above_min]
        below_min = [j for j in range(len(mu_p_star)) if mu_p_star[j] < mu_min]
        unefficient_mu = [mu_p_star[j] for j in below_min]
        unefficient_sigma = [sigma_p[j] for j in below_min]

        alpha = np.linspace(-0.5, 1.5, 10000)
        Z_mu = []
        Z_sigma = []
        for i in range(len(alpha)):
            temp_1 = alpha[i] * mu_min + (1 - alpha[i]) * (mu_p_star[-1])
            Z_mu.append(temp_1)
            temp_2 = self.g + temp_1 * self.h
            Z_sigma.append(np.sqrt(self.threeMatrixDot(np.transpose(temp_2), self.Omega, temp_2))[0][0])

        #print(Z_mu)
        #print(Z_sigma)

        above_min = [i for i in range(len(Z_mu)) if Z_mu[i] > mu_min]
        efficient_mu = [Z_mu[i] for i in above_min]
        efficient_sigma = [Z_sigma[i] for i in above_min]
        below_min = [j for j in range(len(Z_mu)) if Z_mu[j] < mu_min]
        unefficient_mu = [Z_mu[j] for j in below_min]
        unefficient_sigma = [Z_sigma[j] for j in below_min]
        #
        self.efficient_mu = efficient_mu
        self.efficient_sigma = efficient_sigma
        self.unefficient_mu = unefficient_mu
        self.unefficient_sigma = unefficient_sigma
        #
        # self.mu_p_star = mu_p_star


        ### write csv
        #my_file = open("Original_ef_data.csv", 'w')
        #my_data = [efficient_sigma, efficient_mu, unefficient_sigma, unefficient_mu, [sd_min], [mu_min]]
        #with my_file:
            #writer = csv.writer(my_file)
            #writer.writerows(my_data)
        # file.write("efficient_sigma: " + str(efficient_sigma) + '\n');
        # file.write("efficient_mu: " + str(efficient_mu));
        # file.write("unefficient_sigma: " + str(unefficient_sigma) + '\n');
        # file.write("unefficient_mu: " + str(unefficient_mu));
        # file.write("sd_min: " + str(sd_min));
        # file.write("mu_min: " + str(mu_min));
        #file.close()

        ### Plot
        # plt.plot(efficient_sigma, efficient_mu, '-', color = "black", linewidth=2)
        # plt.plot(unefficient_sigma, unefficient_mu, '--', color="black")
        # plt.scatter(sd_min, mu_min, s = 300, color = "red", marker='*')
        # plt.ylabel('Expected portfolio returns')
        # plt.xlabel('Standard deviation of return')
        # plt.title('Assets')
        # plt.show()


    def get_tangencyPortfolio(self, mu_f):
        self.mu_f = mu_f
        ones = np.ones([len(self.Omega), 1])
        self.omega_bar = np.dot(np.linalg.inv(self.Omega), self.mu_vector - self.mu_f*ones)
        self.omega_T = self.omega_bar / np.dot(np.transpose(ones), self.omega_bar)
        self.mu_T = np.dot(np.transpose(self.mu_vector), self.omega_T)[0][0]
        self.sigma_T = np.sqrt(self.threeMatrixDot(np.transpose(self.omega_T),
                                                   self.Omega,
                                                   self.omega_T))

        #print(self.mu_min)
        #print(self.mu_T)
        #print(self.sigma_T)
        #print(self.mu_min)
        #print(self.omega_T)
        #print(self.mu_T)
        #good_mu = [item for item in self.mu_vector]

        # weights_list = []
        # sigma_list = []
        # for i in range(len(self.mu_p_star)):
        #     omega_star = self.g + self.mu_p_star[i] * self.h
        #     weights_list.append(omega_star)
        #     sigma_list.append(np.sqrt(self.threeMatrixDot(np.transpose(omega_star), self.Omega, omega_star))[0][0])
        # self.mu_p_star = np.asarray(self.mu_p_star)
        # sigma_list = np.asarray(sigma_list)
        # sharpe_ratio = (self.mu_p_star - self.mu_f) / sigma_list
        # max_ratio = max(sharpe_ratio)
        # index = np.nonzero(sharpe_ratio == max_ratio)[0][0]
        # self.mu_T = self.mu_p_star[index]
        # self.sigma_T = sigma_list[index]
        # self.omega_T = weights_list[index]
        # print(index)
        # #print(weights_list[index])
        # print("-----")
        # print(max_ratio )






        weights = self.omega_T
        #print(weights)
        weights = [item[0] for item in weights]
        Xaxis = np.arange(len(weights))
        # plt.bar(Xaxis, weights)
        # plt.ylabel('Weights')
        # plt.title('Assets')
        # plt.show()

        # plt.plot(self.efficient_sigma, self.efficient_mu, '-', color="black", linewidth=2)
        # plt.plot(self.unefficient_sigma, self.unefficient_mu, '--', color="black")
        # plt.scatter(self.sd_min, self.mu_min, s=300, color="red", marker='*')
        # plt.scatter(0.5*self.sigma_T, 0.5*(self.mu_T - self.mu_f) + self.mu_f, s=300, color="green", marker='*')
        # plt.plot([0, self.sigma_T], [mu_f, self.mu_T])
        # plt.ylabel('Expected portfolio returns')
        # plt.xlabel('Standard deviation of return')
        # plt.title('Assets')
        #plt.show()

        self.predic = 0.5*(self.mu_T - self.mu_f)

        name = ("AAL", "AAPL", "BABA", "BIDU", "BURL", "D", "DAL",
                         "FB", "FDX", "FRT", "GNC", "GOOG", "GPRO", "L",
                         "NKE",  "O", "S", "SPWR", "T", "UPS")

        self.num_stocks = []
        self.stock_prices = [47.69, 162.75, 175.9, 237.2, 90.95, 80.42, 50.46, 177.48,
                             230.3, 121.37, 6.91, 1026.23, 9.96, 47.99, 55.9, 53.98, 6.97,
                             7.01, 33.92, 121.25]


        ## Calculate the number of each stock
        # test = []
        # for i in range(20):
        #     #MK_price = float(input("Input stock price: "))
        #     MK_price = self.stock_prices[i]
        #     number = 500000 * self.omega_T[i] / MK_price
        #     test.append(number)
        #     self.num_stocks.append(number[0])
        #     print(name[i] + " weight: " + str(self.omega_T[i]) + "； " + str(self.num_stocks[i]))

        ## Plot
        # weights = self.omega_T
        # weights = [item[0] for item in weights]
        # names = ("AAL", "AAPL", "BABA", "BIDU", "BURL", "D", "DAL",
        #          "FB", "FDX", "FRT", "GNC", "GOOG", "GPRO", "L",
        #          "NKE", "O", "S", "SPWR", "T", "UPS")
        # Xaxis = np.arange(len(names))
        # plt.bar(Xaxis, weights)
        # plt.xticks(Xaxis, names)
        # plt.ylabel('Weights')
        # plt.title('Assets')
        # plt.show()






## For 20 assets
#solution = efficientFrontier("AAL.csv", "AAPL.csv", "BABA.csv", "BIDU.csv", "BURL.csv", "D.csv", "DAL.csv",
#                         "FB.csv", "FDX.csv", "FRT.csv", "GNC.csv", "GOOG.csv", "GPRO.csv", "L.csv",
#                         "NKE.csv",  "O.csv", "S.csv", "SPWR.csv", "T.csv", "UPS.csv")
#solution.get_price_list(3)
#solution.get_expected_return()
#solution.set_parameters()
#solution.optimalWeights()
#plt.show()
#solution.get_tangencyPortfolio(0.002)
#solution.save_list("10_27_17")
#plt.show()








 # mu_vector = np.asarray(self.mu_vector)
 #        Omega = np.asarray(self.Omega)
 #
 #        #mu_vector = np.asarray([[0.07], [0.12], [0.09]])
 #        #Omega = np.asarray([[0.2,0,0],[0,0.3,0],[0,0,0.25]])
 #
 #        if np.linalg.det(Omega) == 0:
 #            print("Omega is not reversible !!!")
 #            return
 #        ones = np.ones([len(Omega), 1])
 #        A = self.threeMatrixDot(np.transpose(ones), np.linalg.inv(Omega), mu_vector)[0][0]
 #        B = self.threeMatrixDot(np.transpose(mu_vector), np.linalg.inv(Omega), mu_vector)[0][0]
 #        C = self.threeMatrixDot(np.transpose(ones), np.linalg.inv(Omega), ones)[0][0]
 #        D = np.dot(B, C) - np.dot(A, A)
 #
 #        self.g = (B * np.dot(np.linalg.inv(Omega), ones) - A  * np.dot(np.linalg.inv(Omega), mu_vector)) / D
 #        self.h = (C * np.dot(np.linalg.inv(Omega), mu_vector) - A * np.dot(np.linalg.inv(Omega), ones)) / D












