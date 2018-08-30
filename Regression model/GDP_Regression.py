import numpy as np
import pandas as pd
from prettytable import PrettyTable
import statsmodels.api as sm
from statsmodels.formula.api import ols

import scipy.stats
import matplotlib.pyplot as plt

class GDP:
    def __init__(self, *files):
        self.files = files
        self.value = []
        self.predict = []
        self.beta = []
        self.Y = []
        self.predict_Y = []
        self.H = []


    def read_file(self):
        for file in self.files:
            values = []
            predict = []
            csv = pd.read_csv(file)
            csv.drop(['Security'], axis=1, inplace=True)
            list = csv.as_matrix()[5:]
            list = list[::-1]
            for i in range(len(list) - 3):
                values.append(float(list[i][0]))
            values = np.asarray(values)
            values = np.delete(values, [11])
            self.value.append(values)
            for j in range(3):
                predict.append(float(list[j - 3][0]))
            self.predict.append(predict)
        self.value = np.asarray(self.value)
        self.predict = np.asarray(self.predict)
        #print(self.value[1])

    def calCoeff(self, response, predictors, *delete):
        if len(delete) == 0:
            X = []

            ## Construct X
            for i in range(len(predictors)):
                X.append(predictors[i])
            X = np.transpose(X)
            X = np.concatenate((np.ones([len(X), 1]), X), axis=1)

            ## Construct Y
            Y = np.transpose(response)
            self.Y = Y

            ## Calculate beta
            beta = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)), Y)
            self.beta = beta

            ## Calculate Hat H
            self.H = np.dot(np.dot(X, np.linalg.inv(np.dot(np.transpose(X), X))), np.transpose(X))

            ## Calculate predict Y
            self.predict_Y = np.dot(X, beta)
        else:
            X = []

            ## Construct X
            predictors = np.delete(predictors, delete, axis = 1)
            for i in range(len(predictors)):
                X.append(predictors[i])
            X = np.transpose(X)
            X = np.concatenate((np.ones([len(X), 1]), X), axis=1)

            ## Construct Y
            response = np.delete(response, delete)
            Y = np.transpose(response)
            self.Y = Y

            ## Calculate beta
            beta = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)), Y)
            self.beta = beta

            ## Calculate Hat H
            self.H = np.dot(np.dot(X, np.linalg.inv(np.dot(np.transpose(X), X))), np.transpose(X))

            ## Calculate predict Y
            self.predict_Y = np.dot(X, beta)

    def all_predictors_model(self):
        predictors = self.value[1:]
        response = self.value[0]
        self.calCoeff(response, predictors)
        print("beta_0: " + str(self.beta[0]))
        print("beta_1: " + str(self.beta[1]))
        print("beta_2: " + str(self.beta[2]))
        print("beta_3: " + str(self.beta[3]))
        print("beta_4: " + str(self.beta[4]))
        print("beta_5: " + str(self.beta[5]))


    def ANOVA_mannual(self):
        #predictors = self.value[1:]
        #response = self.value[0]
        ## PickedModel
        predictors = self.value[1:]
        predictors = np.delete(predictors, [0, 1, 4], axis=0)
        response = self.value[0]
        self.calCoeff(response, predictors)
        total_SS = sum((self.Y - np.mean(self.Y))**2)
        regression_SS = sum((self.predict_Y - np.mean(self.Y))**2)
        error_SS = sum((self.Y - self.predict_Y)**2)
        error_MS = error_SS / (len(self.Y) - len(self.beta) + 1 - 1)
        total_MS = total_SS / (len(self.Y) - 1)
        r_squared = 1 - error_SS / total_SS
        adj_r_squared = 1 - error_MS / total_MS
        print(adj_r_squared)






    def ANOVA_innerFc(self):
        df = pd.DataFrame({'GDP': self.value[0],
                           'CPI': self.value[1],
                           'PPI': self.value[2],
                           'us_consumer': self.value[3],
                           'interest_rate': self.value[4],
                           'unemployment': self.value[5]})
        model = ols('GDP ~ CPI+PPI+us_consumer+interest_rate+unemployment', data = df).fit()
        aov_table = sm.stats.anova_lm(model, typ = 2)
        print(aov_table)


    def VIF(self):
        VIF_all = []
        for i in range(len(self.value) - 1):
            predictors = self.value[1:]
            response = predictors[i]
            predictors = np.delete(predictors, i, axis = 0)
            self.calCoeff(response, predictors)
            total_SS = sum((self.Y - np.mean(self.Y)) ** 2)
            error_SS = sum((self.Y - self.predict_Y) ** 2)
            r_squared = 1 - error_SS / total_SS
            VIF_all.append(1/(1 - r_squared))
        print("CPI: " + str(VIF_all[0]))
        print("PPI: " + str(VIF_all[1]))
        print("us_consume: " + str(VIF_all[2]))
        print("interest_rate: " + str(VIF_all[3]))
        print("unemployment: " + str(VIF_all[4]))

    def Cp(self):
        adj_r_squared = ["adj_r^2"]
        C_p_all = ["C_p"]
        ## Using only us_consumer predictor
        predictors = self.value[1:]
        predictors = np.delete(predictors, [0, 1], axis=0)
        response = self.value[0]
        self.calCoeff(response, predictors)
        residual = self.Y - self.predict_Y
        variance_epsilon = sum((residual - np.mean(residual)) ** 2) / (len(self.Y) - len(predictors) - 1)
        predictors = np.asarray([self.value[3]])
        response = self.value[0]
        self.calCoeff(response, predictors)
        error_SS = sum((self.Y - self.predict_Y) ** 2)
        C_p = error_SS/variance_epsilon - len(self.Y) + 2*(len(predictors) + 1)
        C_p_all.append(C_p)
        # R^2
        error_MS = error_SS / (len(self.Y) - len(self.beta) + 1 - 1)
        total_SS = sum((self.Y - np.mean(self.Y)) ** 2)
        total_MS = total_SS / (len(self.Y) - 1)
        adj_r_squared.append(1 - error_MS / total_MS)

        ## Using only unemployment rate predictor
        predictors = self.value[1:]
        predictors = np.delete(predictors, [0, 1], axis=0)
        response = self.value[0]
        self.calCoeff(response, predictors)
        residual = self.Y - self.predict_Y
        variance_epsilon = sum((residual - np.mean(residual)) ** 2) / (len(self.Y) - len(predictors) - 1)
        predictors = np.asarray([self.value[5]])
        response = self.value[0]
        self.calCoeff(response, predictors)
        error_SS = sum((self.Y - self.predict_Y) ** 2)
        C_p = error_SS / variance_epsilon - len(self.Y) + 2 * (len(predictors) + 1)
        C_p_all.append(C_p)
        # R^2
        error_MS = error_SS / (len(self.Y) - len(self.beta) + 1 - 1)
        total_SS = sum((self.Y - np.mean(self.Y)) ** 2)
        total_MS = total_SS / (len(self.Y) - 1)
        adj_r_squared.append(1 - error_MS / total_MS)

        ## Using only interest rate predictor
        predictors = self.value[1:]
        predictors = np.delete(predictors, [0, 1], axis=0)
        response = self.value[0]
        self.calCoeff(response, predictors)
        residual = self.Y - self.predict_Y
        variance_epsilon = sum((residual - np.mean(residual)) ** 2) / (len(self.Y) - len(predictors) - 1)
        predictors = np.asarray([self.value[4]])
        response = self.value[0]
        self.calCoeff(response, predictors)
        error_SS = sum((self.Y - self.predict_Y) ** 2)
        C_p = error_SS / variance_epsilon - len(self.Y) + 2 * (len(predictors) + 1)
        C_p_all.append(C_p)
        # R^2
        error_MS = error_SS / (len(self.Y) - len(self.beta) + 1 - 1)
        total_SS = sum((self.Y - np.mean(self.Y)) ** 2)
        total_MS = total_SS / (len(self.Y) - 1)
        adj_r_squared.append(1 - error_MS / total_MS)

        ## Using us_consume and umemployment
        predictors = self.value[1:]
        predictors = np.delete(predictors, [0, 1], axis = 0)
        response = self.value[0]
        self.calCoeff(response, predictors)
        residual = self.Y - self.predict_Y
        variance_epsilon = sum((residual - np.mean(residual)) ** 2) / (len(self.Y) - len(predictors) - 1)
        predictors = self.value[1:]
        predictors = np.delete(predictors, [0, 1, 3], axis=0)
        response = self.value[0]
        self.calCoeff(response, predictors)
        error_SS = sum((self.Y - self.predict_Y) ** 2)
        C_p = error_SS / variance_epsilon - len(self.Y) + 2 * (len(predictors) + 1)
        C_p_all.append(C_p)
        # R^2
        error_MS = error_SS / (len(self.Y) - len(self.beta) + 1 - 1)
        total_SS = sum((self.Y - np.mean(self.Y)) ** 2)
        total_MS = total_SS / (len(self.Y) - 1)
        adj_r_squared.append(1 - error_MS / total_MS)

        ## Using us_consume and interest_rate
        predictors = self.value[1:]
        predictors = np.delete(predictors, [0, 1], axis=0)
        response = self.value[0]
        self.calCoeff(response, predictors)
        residual = self.Y - self.predict_Y
        variance_epsilon = sum((residual - np.mean(residual)) ** 2) / (len(self.Y) - len(predictors) - 1)
        predictors = self.value[1:]
        predictors = np.delete(predictors, [0, 1, 4], axis=0)
        response = self.value[0]
        self.calCoeff(response, predictors)
        error_SS = sum((self.Y - self.predict_Y) ** 2)
        C_p = error_SS / variance_epsilon - len(self.Y) + 2 * (len(predictors) + 1)
        C_p_all.append(C_p)
        # R^2
        error_MS = error_SS / (len(self.Y) - len(self.beta) + 1 - 1)
        total_SS = sum((self.Y - np.mean(self.Y)) ** 2)
        total_MS = total_SS / (len(self.Y) - 1)
        adj_r_squared.append(1 - error_MS / total_MS)

        ## Using unemployment and interest_rate
        predictors = self.value[1:]
        predictors = np.delete(predictors, [0, 1], axis=0)
        response = self.value[0]
        self.calCoeff(response, predictors)
        residual = self.Y - self.predict_Y
        variance_epsilon = sum((residual - np.mean(residual)) ** 2) / (len(self.Y) - len(predictors) - 1)
        predictors = self.value[1:]
        predictors = np.delete(predictors, [0, 1, 2], axis=0)
        response = self.value[0]
        self.calCoeff(response, predictors)
        error_SS = sum((self.Y - self.predict_Y) ** 2)
        C_p = error_SS / variance_epsilon - len(self.Y) + 2 * (len(predictors) + 1)
        C_p_all.append(C_p)
        # R^2
        error_MS = error_SS / (len(self.Y) - len(self.beta) + 1 - 1)
        total_SS = sum((self.Y - np.mean(self.Y)) ** 2)
        total_MS = total_SS / (len(self.Y) - 1)
        adj_r_squared.append(1 - error_MS / total_MS)

        x = PrettyTable(
            ["Value", "us_consume", "unemployment", "interest_rate", "1*2", "1*3", "2*3"])
        x.align["Value"] = "l"
        x.padding_width = 1
        x.add_row(C_p_all)
        x.add_row(adj_r_squared)
        print(x)

    def leverage(self):
        predictors = self.value[1:]
        response = self.value[0]
        self.calCoeff(response, predictors)
        H_ii = [self.H[i][i] for i in range(len(self.Y))]
        Xaxis = np.arange(len(H_ii))
        plt.scatter(Xaxis, H_ii)

        leverage_X = [j for j in range(len(H_ii)) if H_ii[j] > 2*(len(predictors) + 1) / len(self.Y)]
        leverage_Y = [H_ii[j] for j in range(len(H_ii)) if H_ii[j] > 2*(len(predictors) + 1) / len(self.Y)]
        labels = ['{0}'.format(i) for i in leverage_X]
        plt.scatter(leverage_X, leverage_Y, edgecolors="red")
        for label, x, y in zip(labels, leverage_X, leverage_Y):
            plt.annotate(
                label,
                xy=(x, y), xytext=(-20, 20),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        plt.xlabel("Index")
        plt.ylabel("Lev")
        #plt.hist(H_ii)
        plt.show()

    def rawResidual(self):
        predictors = self.value[1:]
        response = self.value[0]
        self.calCoeff(response, predictors)
        residual = self.Y - self.predict_Y
        #plt.scatter(np.arange(len(self.Y)), residual)
        plt.hist(residual)
        plt.xlabel("Index")
        plt.ylabel("Residual")
        plt.show()

    def studentizedResidual(self):
        predictors = self.value[1:]
        response = self.value[0]
        self.calCoeff(response, predictors)
        residual = self.Y-self.predict_Y
        variance_epsilon = sum((residual - np.mean(residual)) ** 2) / (len(self.Y) - len(predictors) - 1)
        H_ii = [self.H[i][i] for i in range(len(self.Y))]
        s_residual = residual / (np.sqrt(variance_epsilon * (1 - np.asarray(H_ii))))
        plt.scatter(np.arange(len(s_residual)), s_residual)
        #plt.hist(s_residual)

        s_residual_X = [i for i in range(len(s_residual)) if abs(s_residual[i]) > 2]
        s_residual_Y = [i for i in s_residual if abs(i) > 2]
        labels = ['{0}'.format(i) for i in s_residual_X]
        plt.scatter(s_residual_X, s_residual_Y, edgecolors="red")
        for label, x, y in zip(labels, s_residual_X, s_residual_Y):
            plt.annotate(
                label,
                xy=(x, y), xytext=(-20, 20),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        plt.xlabel("Index")
        plt.ylabel("Studentized Residual")
        plt.show()

    def exterStudentizedResidual(self):
        predictors = self.value[1:]
        response = self.value[0]
        self.calCoeff(response, predictors)
        residual = self.Y - self.predict_Y
        H_ii = [self.H[i][i] for i in range(len(self.Y))]

        variance_epsilon = []
        ## remove i_th observation
        for i in range(len(self.Y)):
            response = self.value[0]
            self.calCoeff(response, predictors, i)
            new_residual = self.Y - self.predict_Y
            variance_epsilon.append(sum((new_residual - np.mean(new_residual)) ** 2) / (len(self.Y) - len(predictors) - 1))
        e_s_residual = residual / (np.sqrt(variance_epsilon * (1 - np.asarray(H_ii))))
        #plt.hist(e_s_residual)
        plt.scatter(np.arange(len(e_s_residual)), e_s_residual)

        e_s_residual_X = [i for i in range(len(e_s_residual)) if abs(e_s_residual[i]) > 2 ]
        e_s_residual_Y = [i for i in e_s_residual if abs(i) > 2 ]
        labels = ['{0}'.format(i) for i in e_s_residual_X]
        plt.scatter(e_s_residual_X, e_s_residual_Y, edgecolors="red")
        for label, x, y in zip(labels, e_s_residual_X, e_s_residual_Y):
            plt.annotate(
                label,
                xy=(x, y), xytext=(-20, 20),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        plt.xlabel("Index")
        plt.ylabel("Externally Studentized Residual")
        plt.show()

    def cookD(self):
        predictors = self.value[1:]
        response = self.value[0]
        self.calCoeff(response, predictors)
        residual = self.Y - self.predict_Y
        variance_epsilon = sum((residual - np.mean(residual)) ** 2) / (len(self.Y) - len(predictors) - 1)
        self.predict_Y_1 = self.predict_Y
        cook_d = []
        ## Remove i_th observation
        for i in range(len(self.Y)):
            self.calCoeff(response, predictors, i)
            beta = self.beta
            X = []
            ## Construct X
            for i in range(len(predictors)):
                X.append(predictors[i])
            X = np.transpose(X)
            X = np.concatenate((np.ones([len(X), 1]), X), axis=1)
            self.predict_Y_2 = np.dot(X, beta)
            cook_d.append(sum((self.predict_Y_1 - self.predict_Y_2)**2) / ((len(predictors) + 1)*variance_epsilon))
        plt.scatter(np.arange(len(cook_d)), cook_d)
        cook_d_X = [i for i in range(len(cook_d)) if cook_d[i] > 4 / (len(self.Y) - len(predictors) - 1)]
        cook_d_Y = [i for i in cook_d if i > 4 / (len(self.Y) - len(predictors) - 1)]
        labels = ['{0}'.format(i) for i in cook_d_X]
        plt.scatter(cook_d_X, cook_d_Y, edgecolors="red")
        for label, x, y in zip(labels, cook_d_X, cook_d_Y):
            plt.annotate(
                label,
                xy=(x, y), xytext=(-20, 20),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        plt.xlabel("Index")
        plt.ylabel("Cook's Distance")
        plt.show()

    def pickedModel(self):
        predictors = self.value[1:]
        predictors = np.delete(predictors, [0, 1, 4], axis=0)
        response = self.value[0]
        self.calCoeff(response, predictors)
        print("beta_0: " + str(self.beta[0]))
        print("beta_1: " + str(self.beta[1]))
        print("beta_2: " + str(self.beta[2]))


    def predict_(self):
        predictors = self.value[1:]
        predictors = np.delete(predictors, [0, 1, 4], axis=0)
        response = self.value[0]
        self.calCoeff(response, predictors)
        beta = self.beta
        X = []
        ## Construct X
        predictors = self.predict[1:]
        predictors = np.delete(predictors, [0, 1, 4], axis=0)
        for i in range(len(predictors)):
            X.append(predictors[i])
        X = np.transpose(X)
        X = np.concatenate((np.ones([len(X), 1]), X), axis=1)
        self.predict_Y_2 = np.dot(X, beta)
        Xaxis = np.arange(len(self.predict[0]))
        plt.scatter(Xaxis, self.predict[0], label = "Observation")
        plt.plot(Xaxis, self.predict_Y_2, label = "Prediction")
        plt.legend()
        plt.xlabel("Index")
        plt.ylabel("GDP Growth rate")
        plt.show()


run = GDP("GDP_87_17.csv", "CPI.csv", "PPI.csv", "us consumer spending.csv", "us interest rate.csv",
          "us unemployment rate.csv")
run.read_file()
#run.all_predictors_model()
run.ANOVA_mannual()
#run.ANOVA_innerFc()
#run.VIF()
#run.Cp()
#run.leverage()
#run.rawResidual()
#run.studentizedResidual()
#run.exterStudentizedResidual()
#run.cookD()

#run.pickedModel()
#run.predict_()