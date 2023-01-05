import random # библиотека рандома 
import numpy as np # библиотека для мат вычислений 
from sklearn import datasets  # только для загрузки датасета
import threading # библиотека для создания потоков 
import time # библиотека для замеров времени
import matplotlib.pyplot as plt

INPUT_DIM = 4  # число узлов на входе
OUT_DIM = 3  # число узлов на выходе
H_DIM = 10  # число узлов в скрытом слое
ALPHA = 0.0002  # скорость обучения
NUM_EPOCHS = 600  # колличество эпох
BATCH_SIZE = 50  # размер батча
np.random.seed(1999)  # фиксируем знчение для воспроизводимости

# функция нелинейности
def relu(t):
    return np.maximum(t, 0) # возвращение максимум числа 

# функция для превращения произвольного вектора в набор вероятностей
def softmax(t):
    out = np.exp(t)
    return out / np.sum(out) # функция софтмакс exp(x)/SUM(exp(x))

# функция для превращения произвольного вектора в набор вероятностей для батчей
def softmax_batch(t):
    out = np.exp(t)
    return out / np.sum(out, axis=1, keepdims=True)# keepdims=True результат функции оформляется в массив с количеством осей исходного массива

# разреженная кросс энтропия
def sparse_cross_entropy(z, y):
    return -np.log(z[0, y])

# разреженная кросс энтропия для батчей
def sparse_cross_entropy_batch(z, y):
    return -np.log(np.array([z[j, y[j]] for j in range(len(y))]))

# метод горячего кодирования
def to_full(y, num_classes):
    y_full = np.zeros((1, num_classes))# создаем матрицу нулей по кол-ву признаков 
    y_full[0, y] = 1 # заполняем значение признака в соответствии с указанным 
    return y_full

# метод горячего кодирования для батчей
def to_full_batch(y, num_classes):
    y_full = np.zeros((len(y), num_classes))
    for j, yj in enumerate(y):
        y_full[j, yj] = 1
    return y_full

# функция нелинейной активации в точке t1
def relu_deriv(t):
    return (t >= 0).astype(float)


# функция предсказания
def predict(x):
    t1 = x @ W1 + b1
    h1 = relu(t1)
    t2 = h1 @ W2 + b2
    z = softmax_batch(t2)
    return z

arr_pred = []

# функция расчета точности
def calc_accuracy():
    correct = 0
    for x, y in dataset:
        z = predict(x)
        y_pred = np.argmax(z)
        arr_pred.append(y_pred)
        if y_pred == y:
            correct += 1
    acc = correct / len(dataset)
    return acc

iris = datasets.load_iris()
dataset = [(iris.data[i][None, ...], iris.target[i]) for i in range(len(iris.target))]


# Инициализируем параметры случайными значениями
W1 = np.random.rand(INPUT_DIM, H_DIM)
b1 = np.random.rand(1, H_DIM)
W2 = np.random.rand(H_DIM, OUT_DIM)
b2 = np.random.rand(1, OUT_DIM)

# для равномерного распределения
W1 = (W1 - 0.5) * 2 * np.sqrt(1/INPUT_DIM)
b1 = (b1 - 0.5) * 2 * np.sqrt(1/INPUT_DIM)
W2 = (W2 - 0.5) * 2 * np.sqrt(1/H_DIM)
b2 = (b2 - 0.5) * 2 * np.sqrt(1/H_DIM)


loss_arr = []
acc_arr = []


def nn(data, num_thread):
    global W1
    global b1
    global W2
    global b2
    batch_x, batch_y = zip(*data[i* BATCH_SIZE:i * BATCH_SIZE+BATCH_SIZE])
    x = np.concatenate(batch_x, axis=0)
    y = np.array(batch_y)
   
    # прямое распространение
    t1 = x @ W1 + b1
    
    h1 = relu(t1)
    
    t2 = h1 @ W2 + b2

    z = softmax_batch(t2)  # вектор из вероятностей

    E = np.sum(sparse_cross_entropy_batch(z, y))  # ошибка - разреженная кросс энтропия, т.к y - индекс правильного класаа, а не вектор распределений
   
    # обратное распространение
    y_full = to_full_batch(y, OUT_DIM)  # перевод в представление One Hot Encoding
    dE_dt2 = z - y_full

    dE_dW2 = h1.T @ dE_dt2
    dE_db2 = np.sum(dE_dt2, axis=0, keepdims=True)
    
    dE_dh1 = dE_dt2 @ W2.T
    
    dE_dt1 = dE_dh1 * relu_deriv(t1)
    
    dE_dW1 = x.T @ dE_dt1
    dE_db1 = np.sum(dE_dt1, axis=0, keepdims=True)

    
    # обновление весов
    W1 = W1 - ALPHA * dE_dW1
    b1 = b1 - ALPHA * dE_db1
    W2 = W2 - ALPHA * dE_dW2
    b2 = b2 - ALPHA * dE_db2
    loss_arr.append(E)
    
print("Точность:", calc_accuracy())
tic = time.perf_counter()
for ep in range(NUM_EPOCHS):
    random.shuffle(dataset)
    count = len(dataset) // BATCH_SIZE
    tread_arr = []
    for i in range(count):
        thread = threading.Thread(target=nn, args=(dataset,(f'ep={ep}, i={i}',)))
        tread_arr.append(thread)
        # Запускаем экземпляр `thread`
        thread.start()  
    for i in range(count):
    # Дожидаемся все потоки перед переходом на следующую эпоху
        tread_arr[i].join()
toc = time.perf_counter()


    
# print(f"Вычисление заняло {toc - tic:0.4f} секунд")
print("Точность:", calc_accuracy())
plt.plot(loss_arr)
plt.show()
# x = np.array([7.9, 3.1, 7.5, 1.8])
# pred_cls = np.argmax(predict(x))
# class_names = ['1', '2', '3']
# print('Предсказание:', class_names[pred_cls])







