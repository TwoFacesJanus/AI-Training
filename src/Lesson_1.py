import numpy as np


# Сигмоида.
def nonlin(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))



# Набор входных данных. 
# Каждая строка - это тренировочный пример.
# Каждый столбец - входной узел. 
X = np.array([  [0, 0, 1],
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 1]   ])

# Выходные данные.
y = np.array([[0, 0, 1, 1]]).T


# Делаем случайные числа более определенными.
np.random.seed(1)

# Инициализация весов.
syn0 = 2 * np.random.random((3, 1)) - 1

for iter in range(10000):
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))

    l1_error = y - l1
    print("L1 ERROR:", l1_error)

    l1_delta = l1_error * nonlin(l1, True)

    syn0 += np.dot(l0.T, l1_delta)

print("Выходные данные после тренировки: ")
print(l1)
