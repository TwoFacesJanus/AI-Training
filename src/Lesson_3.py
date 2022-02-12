import numpy as np

''' DATA
NAME        WEIGHT      GROWTH      GENDER

Alice         133         65         F
Bob           160         72         M
Charlie       152         70         M
Diana         120         60         F


NAME        WEIGHT(Minus 135)     GROWTH(Minus 66)      GENDER(1 - F, 0 - M)

Alice            -2                     -1                  1
Bob              25                      6                  0
Charlie          17                      4                  0
Diana           -15                     -6                  1

Обычно сдвигают на среднее значение.

'''


def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2 ).mean()


def sigmoid(x):
    # Сигмоидная функция активации: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
    # Производная сигмоиды: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)


class OurNeuralNetwork:
    '''
    - 2 входа ( вес и рост)
    - скрытый слой с 2 нейрононами (h1, h2)
    - выходной слой с 1 нейроном ( o1 )
    '''

    def __init__(self):
        # weight
        self.w1 = np.random.normal() # wei - w1 - h1
        self.w2 = np.random.normal() # growth - w2 - h1
        self.w3 = np.random.normal() # wei - w3 - h2
        self.w4 = np.random.normal() # growth - w4 - h2
        self.w5 = np.random.normal() # h1 - w5 - o1
        self.w6 = np.random.normal() # h2 - w6 - o1

        # bias
        self.b1 = np.random.normal() # h1
        self.b2 = np.random.normal() # h2
        self.b3 = np.random.normal() # o1

    def feedforward(self, x):
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1

    def train(self, data, all_y_trues):
        '''
        - data - массив numpy (n x 2) numpy, n = к-во наблюдений в наборе. 
        - all_y_trues - массив numpy с n элементами.
        Элементы all_y_trues соответствуют наблюдениям в data.
        '''
        learn_rate = 0.1
        epochs = 1000

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)

                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1

                d_L_d_ypred = -2 * (y_true - y_pred)

                #Neuron o1
                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)

                d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)
                
                 # Нейрон h1
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)

                # Нейрон h2
                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)

                # --- Обновляем веса и пороги
                # Нейрон h1
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                # Нейрон h2
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                # Нейрон o1
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

            # --- Считаем полные потери в конце каждой эпохи
            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(all_y_trues, y_preds)
                print("Epoch %d loss: %.3f" % (epoch, loss))

# Определим набор данных
data = np.array([
  [-2, -1],  # Алиса
  [25, 6],   # Боб
  [17, 4],   # Чарли
  [-15, -6], # Диана
])
all_y_trues = np.array([
  1, # Алиса
  0, # Боб
  0, # Чарли
  1, # Диана
])

# Обучаем нашу нейронную сеть!
network = OurNeuralNetwork()
network.train(data, all_y_trues)


''' NEW DATA FOR TESTS

NAME         WEITH      GROWTH      GENDER
Alice          67         165         F
Bob            90         113         M
Hero           40         190         M
Cursed         20         220         M


NAME         WEIGTH  (minus 54)    GROWTH (minus 172)     GENDER
Alice          13                   -7                    1
Bob            36                   18                    0
Hero           -14                  18                    0
Cursed         6                    28                    0

'''


alice  =  np.array([13, -7])
bob    =  np.array([36, 18])
hero   =  np.array([-14, 18])
cursed =  np.array([6, 28])

print("Alice: %.3f" % network.feedforward(alice))
print("Bob: %.3f" % network.feedforward(bob))
print("Hero: %.3f" % network.feedforward(hero))
print("Cursed: %.3f" % network.feedforward(cursed))
