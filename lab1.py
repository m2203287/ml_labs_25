import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # type: ignore

class Model:
    """Модель парной линейной регрессии с адаптивным обучением"""
    
    def __init__(self, b0=0, b1=0):
        self.b0 = b0
        self.b1 = b1
        
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        return self.b0 + self.b1 * X
    
    def error(self, X, Y):
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        if isinstance(Y, pd.DataFrame):
            Y = Y.iloc[:, 0]
        return np.mean((self.predict(X) - Y)**2) / 2
    
    def fit(self, X, Y, alpha=1.0, min_error_change=1e-6, max_steps=5000):
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        if isinstance(Y, pd.DataFrame):
            Y = Y.iloc[:, 0]
            
        steps, errors = [], []
        step = 0
        prev_error = float('inf')
        
        while step < max_steps:
            dJ0 = np.mean(self.predict(X) - Y)
            dJ1 = np.mean((self.predict(X) - Y) * X)
            
            old_b0, old_b1 = self.b0, self.b1
            self.b0 -= alpha * dJ0
            self.b1 -= alpha * dJ1
            
            current_error = self.error(X, Y)
            
            if current_error > prev_error:
                self.b0, self.b1 = old_b0, old_b1
                alpha /= 2
                continue
                
            error_change = abs(prev_error - current_error)
            if error_change < min_error_change:
                break
                
            prev_error = current_error
            step += 1
            steps.append(step)
            errors.append(current_error)
            
        return steps, errors

    def plot_regression(self, X, Y):
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        if isinstance(Y, pd.DataFrame):
            Y = Y.iloc[:, 0]
            
        plt.figure(figsize=(10, 6))
        plt.scatter(X, Y, alpha=0.5, label='Точки данных')
        
        x_min, x_max = X.min(), X.max()
        x_line = np.linspace(x_min, x_max, 100)
        y_line = self.predict(x_line)
        
        plt.plot(x_line, y_line, 'r', label='Линия регрессии')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Линейная регрессия')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def plot_learning_curve(self, steps, errors):
        plt.figure(figsize=(10, 6))
        plt.plot(steps, errors, 'g-')
        plt.xlabel('Итерации')
        plt.ylabel('Среднеквадратичная ошибка')
        plt.title('Кривая обучения')
        plt.grid(True)
        plt.show()

# Загрузка данных
x = pd.read_csv('https://raw.githubusercontent.com/koroteevmv/ML_course/2023/ML1.1%20linear%20regression/data/x.csv', index_col=0)['0']
y = pd.read_csv('https://raw.githubusercontent.com/koroteevmv/ML_course/2023/ML1.1%20linear%20regression/data/y.csv', index_col=0)['0']

# Создание и обучение модели
model = Model()
steps, errors = model.fit(x, y)
print(f"Итоговая ошибка: {model.error(x, y):.6f}")

# Визуализация результатов
model.plot_regression(x, y)
model.plot_learning_curve(steps, errors)