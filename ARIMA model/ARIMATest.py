import numpy as np
import pandas as pd
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA, ARIMAResults, ARMA
from statsmodels.tsa.stattools import acf
from sklearn.metrics import mean_squared_error

class Solver:
    def __init__(self, file, is_stock):
        self.file = file
        self.is_stock = is_stock
        self.train_set = []
        self.test_set = []
        self.price_list = []
        self.orginal_data = []
        self.manual_predictions = []
        self.model_predictions = []
        self.aic = []
        self.bic = []
        self.model_fit = 0
        self.mu = 0
        self.base_AIC_or_BIC = 0

    def get_price_array(self):
        if self.is_stock:
            csv = pd.read_csv(self.file)
            csv.drop(['Date', 'Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1, inplace=True)
            list = csv.as_matrix()
            for i in range(len(list)):
                self.price_list.append(float(list[i][0]))
                self.orginal_data.append(float(list[i][0]))
        else:
            csv = pd.read_csv(self.file)
            csv.drop(['Security'], axis=1, inplace=True)
            list = csv.as_matrix()[6:]
            for i in range(len(list)):
                self.price_list.append(float(list[i][0]))
                self.orginal_data.append(float(list[i][0]))

    def check_stationary(self):
        for i in range(3):
            price_list = np.diff(self.price_list, n = i)
            acf_set = acf(price_list)
            Xaxis = range(len(acf_set))
            plt.figure(1)
            plt.subplot(1,3,i+1)
            plot = plt.plot(Xaxis, acf_set)
            plt.title("D = " + str(i))
            plt.xlabel("Lag")
            plt.ylabel("ACF")
        plt.show()

    def get_train_test(self, diff):
        if self.is_stock == False:
            self.price_list = np.diff(self.price_list, n=diff)
            index = len(self.price_list[0:-3])
            for i in range(index):
                self.train_set.append(float(self.price_list[i]))
            for j in range(index , len(self.price_list)):
                self.test_set.append(float(self.price_list[j]))
        else:
            returns = []
            for t in range(len(self.price_list) - 1):
                returns.append((self.price_list[t + 1] - self.price_list[t])
                           / self.price_list[t])
            self.price_list = np.log(1 + np.asarray(returns)).tolist()
            #self.price_list = np.diff(self.price_list, n=diff)
            self.train_set = self.price_list[0:-4]
            self.test_set = self.price_list[-4:-1]



    def model_select(self):
        self.aic = []
        self.bic = []
        ## If data is stock
        if self.is_stock:
            for i in range(3):
                for j in range(3):
                    if i == 0:
                        if j == 2:
                            continue
                        init = [.1 for j in range(i + j + 2)]
                        model = ARIMA(self.train_set, order=(j + 1, 0, i))
                        self.model_fit = model.fit(start_params=init,disp=0)

                        self.aic.append(ARIMAResults.aic(self.model_fit))
                        self.bic.append(ARIMAResults.bic(self.model_fit))
                        #print(ARIMAResults.summary(self.model_fit))
                    else:
                        init = [.4 for j in range(i + j + 1)]
                        model = ARIMA(self.train_set, order=(j, 0, i))
                        self.model_fit = model.fit(start_params=init, maxiter = 150,disp=0)

                        self.aic.append(ARIMAResults.aic(self.model_fit))
                        self.bic.append(ARIMAResults.bic(self.model_fit))
                        #print(ARIMAResults.summary(self.model_fit))
        ## If data is forex
        else:
            for i in range(3):
                for j in range(3):
                    if i == 0:
                        if j == 2:
                            continue
                        init = [.1 for j in range(i + j + 2)]
                        model = ARIMA(self.train_set, order=(j + 1, 0, i))
                        self.model_fit = model.fit(start_params=init, maxiter=150,disp=0)

                        self.aic.append(ARIMAResults.aic(self.model_fit))
                        self.bic.append(ARIMAResults.bic(self.model_fit))
                        print(ARIMAResults.summary(self.model_fit))
                    else:
                        init = [.28 for j in range(i + j + 1)]
                        model = ARIMA(self.train_set, order=(j, 0, i))

                        self.model_fit = model.fit(start_params=init, maxiter=150, disp=0)

                        self.aic.append(ARIMAResults.aic(self.model_fit))
                        self.bic.append(ARIMAResults.bic(self.model_fit))
                        print(ARIMAResults.summary(self.model_fit))

        ## Compare AIC and BIC, select the best model
        min_aic = min(self.aic)
        min_bic = min(self.bic)

        x = PrettyTable(["Model", "AIC", "BIC"])
        x.align["Model"] = "l"
        x.padding_width = 1
        x.add_row(["ARIMA(1,1,0)", self.aic[0], self.bic[0]])
        x.add_row(["ARIMA(2,1,0)", self.aic[1], self.bic[1]])
        x.add_row(["ARIMA(0,1,1)", self.aic[2], self.bic[2]])
        x.add_row(["ARIMA(1,1,1)", self.aic[3], self.bic[3]])
        x.add_row(["ARIMA(2,1,1)", self.aic[4], self.bic[4]])
        x.add_row(["ARIMA(0,1,2)", self.aic[5], self.bic[5]])
        x.add_row(["ARIMA(1,1,2)", self.aic[6], self.bic[6]])
        x.add_row(["ARIMA(2,1,2)", self.aic[7], self.bic[7]])
        print(x)
        #print(self.aic.index(min_aic), self.bic.index(min_bic))


    def set_AIC_or_BIC(self, base_AIC_or_BIC):
        self.base_AIC_or_BIC = base_AIC_or_BIC

    def predict(self):
        ## If data is stock
        if self.is_stock:
            p, q = [1, 1]
            init = [.4 for j in range(p + q + 1)]
            self.model = ARIMA(self.train_set, order=(p, 0, q))
            self.model_fit = self.model.fit(start_params=init, maxiter=150, disp=0)
            ar_coe = ARIMAResults.arparams(self.model_fit)
            ma_coe = ARIMAResults.maparams(self.model_fit)
            print(ARIMAResults.summary(self.model_fit))

            error = ARIMAResults.resid(self.model_fit).tolist()

            # Predict n+1
            self.manual_predictions.append(self.mu - ma_coe[0] * error[-1])
            error.append(self.test_set[0] - self.manual_predictions[0])
            # Predict n+2
            self.manual_predictions.append(self.mu - ma_coe[0] * error[-1])
            error.append(self.test_set[1] - self.manual_predictions[1])
            # Predict n+3
            self.manual_predictions.append(self.mu - ma_coe[0] * error[-1])
            error.append(self.test_set[2] - self.manual_predictions[2])

            self.model_predictions = self.model_fit.forecast(3)[0]

        ## If data is forex
        else:
            if self.base_AIC_or_BIC == 'AIC':
                p, q = [2,2]
                init = [.28 for j in range(p + q + 1)]
                self.model = ARIMA(self.train_set, order=(p, 0, q))
                self.model_fit = self.model.fit(start_params=init, maxiter=150, disp=0)
                ar_coe = ARIMAResults.arparams(self.model_fit)
                ma_coe = ARIMAResults.maparams(self.model_fit)
                print(ARIMAResults.summary(self.model_fit))

                error = ARIMAResults.resid(self.model_fit).tolist()

                # Predict n+1
                self.manual_predictions.append(self.mu + ar_coe[0]*(self.train_set[-1] - self.mu)
                                        + ar_coe[1]*(self.train_set[-2] - self.mu)
                                        -ma_coe[0]*error[-1] - ma_coe[1]*error[-2])
                error.append(self.test_set[0] - self.manual_predictions[0])
                # Predict n+2
                self.manual_predictions.append(self.mu + ar_coe[0] * (self.test_set[0] - self.mu)
                                        + ar_coe[1] * (self.train_set[-1] - self.mu)
                                        - ma_coe[0] * error[-1] - ma_coe[1] * error[-2])
                error.append(self.test_set[1] - self.manual_predictions[1])
                # Predict n+3
                self.manual_predictions.append(self.mu + ar_coe[0] * (self.test_set[1] - self.mu)
                                        + ar_coe[1] * (self.test_set[0] - self.mu)
                                        - ma_coe[0] * error[-1] - ma_coe[1] * error[-2])
                error.append(self.test_set[2] - self.manual_predictions[2])

                self.model_predictions = self.model_fit.forecast(3)[0]
            else:
                p, q = [0, 1]
                init = [.28 for j in range(p + q + 1)]
                self.model = ARIMA(self.train_set, order=(p, 0, q))
                self.model_fit = self.model.fit(start_params=init, maxiter=150, disp=0)
                ar_coe = ARIMAResults.arparams(self.model_fit)
                ma_coe = ARIMAResults.maparams(self.model_fit)
                print(ARIMAResults.summary(self.model_fit))

                error = ARIMAResults.resid(self.model_fit).tolist()

                # Predict n+1
                self.manual_predictions.append(self.mu - ma_coe[0] * error[-1])
                error.append(self.test_set[0] - self.manual_predictions[0])
                # Predict n+2
                self.manual_predictions.append(self.mu - ma_coe[0] * error[-1])
                error.append(self.test_set[1] - self.manual_predictions[1])
                # Predict n+3
                self.manual_predictions.append(self.mu - ma_coe[0] * error[-1])
                error.append(self.test_set[2] - self.manual_predictions[2])

                self.model_predictions = self.model_fit.forecast(3)[0]

    def predict_original_value(self):
        original_test = self.orginal_data[-4:-1]
        original_manual_predictions = []
        original_model_predictions = []

        original_manual_predictions.append(self.orginal_data[-4] +
                                           self.manual_predictions[0])
        original_manual_predictions.append(self.orginal_data[-4] +
                                           self.manual_predictions[0] +
                                           self.manual_predictions[1])
        original_manual_predictions.append(self.orginal_data[-4] +
                                           self.manual_predictions[0] +
                                           self.manual_predictions[1] +
                                           self.manual_predictions[2])

        original_model_predictions.append(self.orginal_data[-4] +
                                           self.model_predictions[0])
        original_model_predictions.append(self.orginal_data[-4] +
                                           self.model_predictions[0] +
                                           self.model_predictions[1])
        original_model_predictions.append(self.orginal_data[-4] +
                                           self.model_predictions[0] +
                                           self.model_predictions[1] +
                                           self.model_predictions[2])
        plt.plot(original_test, color='red', label="validation data")
        plt.plot(original_manual_predictions, color='blue', label="manual predictions")
        plt.plot(original_model_predictions, color='green', label="model predictions")
        plt.xlabel("time")
        plt.ylabel("Ratio")
        plt.legend()
        plt.show()

    def set_mu(self, mu):
        self.mu = mu

    def mse(self, predictions):
        return mean_squared_error(self.test_set, predictions)


##################### AAPL #####################
# AAPL = Solver("AAPL.csv", True)
# AAPL.get_price_array()
#
# AAPL.check_stationary()
# AAPL.get_train_test(1)
# AAPL.model_select()
# AAPL.set_mu(0.0007)
# AAPL.predict()
#
# plt.plot(AAPL.test_set, color = 'red',label = "validation data")
# plt.plot(AAPL.manual_predictions, color = 'blue', label = "manual predictions")
# plt.plot(AAPL.model_predictions, color = 'green', label = "model predictions")
# plt.xlabel("time")
# plt.ylabel("Difference of daily log return")
# plt.legend()
# print("Manual MSE: " + str(AAPL.mse(AAPL.manual_predictions)))
# print("Model MSE: " + str(AAPL.mse(AAPL.model_predictions)))
# print("Mean value of validation data: " + str(np.mean(AAPL.test_set)))
# plt.show()



##################### FOREX #####################
# FOREX = Solver("GBP_USD_FOREX.csv", False)
# FOREX.get_price_array()
#
# FOREX.check_stationary()
# FOREX.get_train_test(1)
# FOREX.model_select()
# FOREX.set_AIC_or_BIC("BIC")
# FOREX.set_mu(0.0003)
# FOREX.predict()
# # plt.plot(FOREX.test_set, color = 'red',label = "validation data")
# # plt.plot(FOREX.manual_predictions, color = 'blue', label = "manual predictions")
# # plt.plot(FOREX.model_predictions, color = 'green', label = "model predictions")
# # plt.xlabel("time")
# # plt.ylabel("Ratio")
# # plt.legend()
# # print("Manual MSE: " + str(FOREX.mse(FOREX.manual_predictions)))
# # print("Model MSE: " + str(FOREX.mse(FOREX.model_predictions)))
# # print("Mean value of validation data: " + str(np.mean(FOREX.test_set)))
# # plt.show()

#FOREX.predict_original_value()

