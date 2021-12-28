import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Сгенерируем датасет
def f(x):
    return np.exp(x)
x = np.linspace(-4, 10, 500)
y = f(x)

#Здесь мы будем хранить результаты апроксимации.
# Лучшее рещение выбирается исходя из метрики r2_score
result = []

# Разобъём датасет на обучающую и тестовую выборки
x = [[x[i], i] for i in range(len(x))]
y = [[y[i], i] for i in range(len(y))]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
x_train.sort(key=lambda i: i[1])
y_train.sort(key=lambda i: i[1])
x_test.sort(key=lambda i: i[1])
y_test.sort(key=lambda i: i[1])
x_train = np.array([x_train[i][0] for i in range(len(x_train))])
y_train = np.array([y_train[i][0] for i in range(len(y_train))])
x_test = np.array([x_test[i][0] for i in range(len(x_test))])
y_test = np.array([y_test[i][0] for i in range(len(y_test))])

# Создадим график
fig, ax = plt.subplots()

# Добавим на него обучающие точки
ax.scatter(x_train, y_train, label="Тренировачные точки")

# Пройдём в цикле по полиномам степеней 1-5
for degree in range(1, 6):
    # Подберём коэффициенты полинома под обучающую выборку 
    model = np.polyfit(x_train, y_train, degree)[::-1]
    # Построим функцию предсказатель
    polapprox = lambda x: sum([model[i]*x**i for i in range(0, degree + 1)])
    # Зададим предсказанный график полинома с помощью предсказателя
    ax.plot(x_test, polapprox(x_test), label=f"Полином степени {degree}")
    # Внесём данные в результат
    result.append([r2_score(polapprox(x_test), y_test),
                    f"Полином степени {degree}",
                    model[::-1]])

# Подберём коэффициенты показательного урравнения регрессии под обучающую выборку
b, a = map(float, np.polyfit(x_train, np.log(y_train), 1))
# Построим функцию предсказатель
expapprox = lambda x: np.exp(a) * np.exp(b*x)
# Зададим предсказанный график уравения полиномиальной регрессии с помощью предсказателя
plt.plot(x_test, expapprox(x_test), label="Показательное уравнение регрессии")
# Внесём даные в результат
result.append([r2_score(expapprox(x_test), y_test),
                    "Показательное уравнение регрессии",
                    [a, b]])

# Добавим легенду на график
ax.legend()
# Выведем  результаты
print("Лучший результат:\n", max(result, key= lambda x: x[0]))
print("Полные результаты:")
for arr in result:
    print(arr)
plt.show()