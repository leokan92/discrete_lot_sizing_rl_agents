# -*- coding: utf-8 -*-
import math
import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


class RegressorLib():
    def __init__(self, model_name, env):
        super(RegressorLib, self).__init__()
        self.env = env
        self.model_name = model_name
        self.has_been_fitted = False
        if self.model_name == 'Random Forest':
            self.regressor = RandomForestRegressor(n_estimators=50)
        elif self.model_name == 'Linear Regression':
            self.regressor = LinearRegression()
        elif self.model_name == 'Decision Tree':
            self.regressor = DecisionTreeRegressor(
                random_state=0
            )
        elif self.model_name == 'plain_matrix_I2xM1':
            self.V = [
                np.zeros(shape=(env.max_inventory_level[0] + 1, env.max_inventory_level[1] + 1)),
                np.zeros(shape=(env.max_inventory_level[0] + 1, env.max_inventory_level[1] + 1)),
                np.zeros(shape=(env.max_inventory_level[0] + 1, env.max_inventory_level[1] + 1)),
            ]
        elif self.model_name == 'matrix_independent':
            self.V = []
            # inventory contribution
            for i in range(self.env.n_items):
                self.V.append(
                    np.zeros(
                        shape=(
                            env.max_inventory_level[i] + 1 # 0 item, ..., max_inv_level
                        )
                    )
                )
            # set up contribution
            for i in range(self.env.n_items):
                self.V.append(
                    np.zeros(
                        shape=(
                            env.n_machines + 1 # no machine, 1 machine, ... n_machines
                        )
                    )
                )

    def fit(self, X_train, y_train):
        alpha = 1 / self.env.T
        if self.model_name == 'plain_matrix_I2xM1':
            if X_train[2] == 1:
                idx = 0
            elif X_train[3] == 1:
                idx = 1
            elif X_train[4] == 1:
                idx = 2
            # self.V[idx][int(X_train[0]), int(X_train[1])] = y_train
            # NEL Q-LEARNING 
            old = self.V[idx][int(X_train[0]), int(X_train[1])]
            self.V[idx][int(X_train[0]), int(X_train[1])] = (1-alpha) * old + alpha * y_train
        elif self.model_name == 'matrix_independent':
            # suppose that the value is due to all the items in the same way
            new_val = y_train / (self.env.n_items)

            for i in range(self.env.n_items):
                old = self.V[i][int(X_train[i])]
                self.V[i][int(X_train[i])] = (1-alpha) * old + alpha * new_val
                # self.V[i][int(X_train[i])] = (1-alpha) * old + alpha * 0.8 * new_val
            # for i in range(self.env.n_items):
            #     old = self.V[i + self.env.n_items][X_train[i + self.env.n_items]]
            #     # self.V[i + self.env.n_items][machine_state[i]] = (1-alpha) * old + alpha*0.2 * y_train / (2 * self.env.n_items)
            #     self.V[i + self.env.n_items][X_train[i + self.env.n_items]] = (1-alpha) * old + alpha * 0.2 * new_val

        else:
            # RESTART FROM ZERO
            self.regressor = RandomForestRegressor(n_estimators=50)
            self.regressor.fit(X_train, y_train)  # warm_start=False
            print(f"R2: {self.regressor.score(X_train, y_train)}")
            self.has_been_fitted = True

    def predict(self, X_test):
        if self.model_name == 'plain_matrix_I2xM1':
            if X_test[0][2] == 1:
                idx = 0
            elif X_test[0][3] == 1:
                idx = 1
            elif X_test[0][4] == 1:
                idx = 2
            return self.V[idx][int(X_test[0][0]), int(X_test[0][1])]
        elif self.model_name == 'matrix_independent':
            ans = 0
            # inv contribution:
            for i in range(self.env.n_items):
                ans += self.V[i][int(X_test[0][i])]
            # setup contribution:
            for i in range(self.env.n_items):
                ans += self.V[i + self.env.n_items][int(X_test[0][i + self.env.n_items])]
            return ans

        if self.has_been_fitted:
            output = self.regressor.predict(X_test).item()
        else:
            # e.g. in the first round of the iterations
            output = 0
        return output

    def evaluate(self, X_test, y_test):
        y_predict = self.regressor.predict(X_test)
        error = abs(y_predict - y_test)
        print(f"R2: {r2_score(y_test, self.regressor.predict(X_test)):.2f}")
        print(f"mean erro: {np.mean(error):.2f}")
