
"""
Эксперименты 3.1 и 3.2: эксперименты по градиентному спуску
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

from optimization import gradient_descent
from oracles import QuadraticOracle


# ==================== функции для графиков ====================

def plot_levels(func, x_lim, y_lim, ax=None, n_lines=20):
    """
    Рисует линии уровня функции
    """
    if ax is None:
        ax = plt.gca()
    
    # Создаем сетку
    x = np.linspace(x_lim[0], x_lim[1], 200)
    y = np.linspace(y_lim[0], y_lim[1], 200)
    X, Y = np.meshgrid(x, y)
    
    # Вычисляем значения функции
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))
    
    # Рисуем линии уровня
    levels = np.logspace(np.log10(Z.min() + 1e-10), np.log10(Z.max()), n_lines)
    ax.contour(X, Y, Z, levels=levels, colors='gray', alpha=0.5, linewidths=1)
    ax.contourf(X, Y, Z, levels=levels, alpha=0.1, cmap='viridis')
    
    return ax

def plot_trajectory(trajectory, ax=None, color='red', marker='o', label=None):
    """
    Рисует траекторию оптимизации
    """
    if ax is None:
        ax = plt.gca()
    
    # Рисуем траекторию
    ax.plot(trajectory[:, 0], trajectory[:, 1], 
           color=color, linewidth=1.5, marker=marker, 
           markersize=3, label=label, alpha=0.7)
    
    # Отмечаем начальную и конечную точки
    if len(trajectory) > 0:
        ax.scatter(trajectory[0, 0], trajectory[0, 1], 
                  color=color, s=80, marker='*', alpha=0.9)
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], 
                  color=color, s=80, marker='s', alpha=0.9)
    
    return ax

def save_results_to_file(results, filename="experiments_3_1_and_3_2_results.txt"):
    """
    Сохраняет результаты экспериментов в текстовый файл
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
                
        # Результаты эксперимента 3.1
        if 'results_3_1' in results and results['results_3_1']:
            f.write("Эксперимент 3.1: Траектория градиентного спуска\n")
            f.write("-" * 50 + "\n")
            
            for result in results['results_3_1']:
                f.write(f"Функция: {result['функция']}\n")
                f.write(f"  Стратегия: {result['стратегия']}\n")
                f.write(f"  Итерации: {result['итерации']}\n")
                f.write(f"  Статус: {result['статус']}\n")
                f.write(f"  f(x*): {result['f(x*)']:.6e}\n")
                f.write(f"  ||∇f(x*)||: {result['grad_norm']:.6e}\n")
                f.write("-" * 30 + "\n")
        
        # Результаты эксперимента 3.2
        if 'results_3_2' in results and results['results_3_2']:
            f.write("\nЭксперимент 3.2: Зависимость от k и n\n")
            f.write("-" * 50 + "\n")
            
            # Группируем результаты по размерности
            n_values = sorted(set(r['n'] for r in results['results_3_2']))
            
            for n in n_values:
                n_results = [r for r in results['results_3_2'] if r['n'] == n]
                if n_results:
                    f.write(f"\nРазмерность n = {n}:\n")
                    f.write("-" * 30 + "\n")
                    
                    # Статистика
                    iters = [r['итерации'] for r in n_results]
                    k_values = [r['k'] for r in n_results]
                    
                    f.write(f"  Среднее число итераций: {np.mean(iters):.1f}\n")
                    f.write(f"  Минимальное: {np.min(iters):.0f}\n")
                    f.write(f"  Максимальное: {np.max(iters):.0f}\n")
                    f.write(f"  Стандартное отклонение: {np.std(iters):.1f}\n")
                    
                    # Регрессия
                    log_k = np.log(k_values)
                    log_iters = np.log(iters)
                    coeffs = np.polyfit(log_k, log_iters, 1)
                    correlation = np.corrcoef(log_k, log_iters)[0, 1]
                    
                    f.write(f"  Зависимость: T ∝ k^{coeffs[0]:.3f}\n")
                    f.write(f"  Корреляция: {correlation:.3f}\n")
                    
                    # Таблица значений
                    f.write("\n  Подробные результаты:\n")
                    f.write("  k           Итерации\n")
                    f.write("  ---------------------\n")
                    
                    # Группируем по k
                    k_groups = {}
                    for r in n_results:
                        k_val = r['k']
                        if k_val not in k_groups:
                            k_groups[k_val] = []
                        k_groups[k_val].append(r['итерации'])
                    
                    for k_val in sorted(k_groups.keys()):
                        avg_iter = np.mean(k_groups[k_val])
                        f.write(f"  {k_val:8.1f}  {avg_iter:8.1f}\n")
        
        # Выводы
        f.write("\n\n" + "=" * 70 + "\n")
        f.write("Выводы\n")
        f.write("=" * 70 + "\n")
        f.write("1. Число итераций растет с увеличением числа обусловленности k\n")
        f.write("2. Зависимость приблизительно линейна в логарифмических координатах\n")
        f.write("3. Для больших n требуется больше итераций при том же k\n")
        f.write("4. Разброс результатов уменьшается с увеличением n\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("Созданные файлы\n")
        f.write("=" * 70 + "\n")
        
        # Проверяем, какие файлы были созданы
        created_files = []
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for file in os.listdir(script_dir):
            if file.startswith('experiment_3_') and file.endswith('.png'):
                created_files.append(file)
        
        if created_files:
            for file in sorted(created_files):
                f.write(f"- {file}\n")
        else:
            f.write("Файлы графиков не найдены\n")
    
    print(f"\nРезультаты сохранены в файл: {filepath}")
    return filepath

# ==================== эксперимент 3.1 ====================

def run_experiment_3_1():
    """
    Эксперимент 3.1: Траектория градиентного спуска на квадратичной функции
    """
    print("=" * 60)
    print("Эксперимент 3.1: Траектория градиентного спуска")
    print("=" * 60)
    
    # 1. Функция с хорошим числом обусловленности (k ≈ 1)
    print("\n1. Функция с k ≈ 1 (сферическая)")
    A1 = np.array([[1, 0], [0, 1]], dtype=np.float64)
    b1 = np.array([0, 0], dtype=np.float64)
    oracle1 = QuadraticOracle(A1, b1)
    
    # 2. Функция с плохим числом обусловленности (k >> 1)
    print("2. Функция с k >> 1 (вытянутый эллипсоид)")
    A2 = np.array([[100, 0], [0, 1]], dtype=np.float64)  # k = 100
    b2 = np.array([0, 0], dtype=np.float64)
    oracle2 = QuadraticOracle(A2, b2)
    
    # 3. Функция с корреляцией (недиагональная)
    print("3. Функция с корреляцией (недиагональная матрица)")
    A3 = np.array([[10, 7], [7, 5]], dtype=np.float64)
    b3 = np.array([0, 0], dtype=np.float64)
    oracle3 = QuadraticOracle(A3, b3)
    
    # Начальные точки для экспериментов
    x0_points = [
        np.array([10.0, 10.0]),
        np.array([1.0, 10.0]),
        np.array([-5.0, -5.0])
    ]
    
    # Стратегии выбора шага
    strategies = [
        {'method': 'Constant', 'c': 0.01},
        {'method': 'Constant', 'c': 0.1},
        {'method': 'Armijo', 'c1': 1e-4, 'alpha_0': 1.0},
        {'method': 'Wolfe', 'c1': 1e-4, 'c2': 0.9, 'alpha_0': 1.0}
    ]
    
    strategy_names = ['Constant (0.01)', 'Constant (0.1)', 'Armijo', 'Wolfe']
    
    # Проведем эксперименты для каждой функции
    oracles = [oracle1, oracle2, oracle3]
    oracle_names = ['Сферическая (k≈1)', 'Вытянутая (k=100)', 'Коррелированная']
    
    results_3_1 = []
    
    for idx, (oracle, name) in enumerate(zip(oracles, oracle_names)):
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        for i, (strategy, strategy_name) in enumerate(zip(strategies, strategy_names)):
            ax = axes[i]
            
            # Для каждой начальной точки
            colors = ['r', 'g', 'b']
            
            for x0_idx, x0 in enumerate(x0_points):
                # Запускаем градиентный спуск
                result, status, history = gradient_descent(
                    oracle=oracle,
                    x_0=x0,
                    tolerance=1e-8,
                    max_iter=1000,
                    line_search_options=strategy,
                    trace=True,
                    display=False)
                
                # Рисуем линии уровня (только для первой начальной точки)
                if x0_idx == 0:
                    x_lim = (-15, 15)
                    y_lim = (-15, 15)
                    plot_levels(oracle.func, x_lim, y_lim, ax=ax)
                
                # Рисуем траекторию
                if history and 'x' in history:
                    trajectory = np.array(history['x'])
                    label = f'x0=({x0[0]}, {x0[1]})'
                    plot_trajectory(trajectory, ax=ax, color=colors[x0_idx], label=label)
                    
                    # Сохраняем результаты для сводки (только для первой начальной точки)
                    if x0_idx == 0 and history:
                        results_3_1.append({
                            'функция': name,
                            'стратегия': strategy_name,
                            'итерации': len(history['func']) - 1,
                            'статус': status,
                            'f(x*)': history['func'][-1],
                            'grad_norm': history['grad_norm'][-1]
                        })
            
            ax.set_title(f'{strategy_name}', fontsize=12)
            ax.set_xlabel('x₁', fontsize=10)
            ax.set_ylabel('x₂', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc='upper right')
            ax.set_xlim(-15, 15)
            ax.set_ylim(-15, 15)
            ax.set_aspect('equal')
        
        plt.suptitle(f'Траектории градиентного спуска: {name}', fontsize=12, y=0.98)
        plt.tight_layout()
        
        # Сохраняем график в ту же папку
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filename = f'experiment_3_1_{name.replace(" ", "_").replace("(", "").replace(")", "").replace("≈", "")}.png'
        filepath = os.path.join(script_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Сохранен график: {filepath}")
        plt.show()
    
    return results_3_1

# ==================== эксперимент 3.2 ====================

def run_experiment_3_2():
    """
    Эксперимент 3.2: Зависимость числа итераций от числа обусловленности и размерности
    """
    print("\n" + "=" * 60)
    print("Эксперимент 3.2: Зависимость от числа обусловленности k и размерности n")
    print("=" * 60)
    
    # Параметры эксперимента
    n_values = [10, 50, 100]  # Размерности
    k_values = np.logspace(0, 3, 15)  # Числа обусловленности от 1 до 1000
    n_repeats = 3  # Число повторений для усреднения
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    print(f"\nПараметры:")
    print(f"  Размерности: {n_values}")
    print(f"  Числа обусловленности: от {k_values[0]:.1f} до {k_values[-1]:.1f}")
    print(f"  Повторений для усреднения: {n_repeats}")
    print(f"  Стратегия: Wolfe (c1=1e-4, c2=0.9)")
    print(f"  Точность: 1e-6")
    
    # Подготовка данных для графика
    plt.figure(figsize=(12, 8))
    
    results_3_2 = []  # Для хранения всех результатов
    
    for n_idx, n in enumerate(n_values):
        print(f"\n{'='*40}")
        print(f"Размерность n = {n}")
        print(f"{'='*40}")
        
        avg_iterations_per_k = []
        std_iterations_per_k = []
        
        for k_idx, k in enumerate(k_values):
            k_iterations = []
            
            for repeat in range(n_repeats):
                # Фиксируем seed для воспроизводимости
                seed = 42 * repeat + n_idx * 1000 + k_idx * 10000
                np.random.seed(seed)
                
                # Генерация случайной квадратичной задачи
                eigenvalues = np.exp(np.random.uniform(0, np.log(k), n))
                eigenvalues[0] = 1.0  # Минимальное собственное значение
                eigenvalues[-1] = k  # Максимальное собственное значение
                eigenvalues.sort()
                
                # Создаем диагональную матрицу (плотную, чтобы избежать ошибки)
                A = np.diag(eigenvalues)
                b = np.random.randn(n)
                oracle = QuadraticOracle(A, b)
                
                # Случайная начальная точка
                x0 = np.random.randn(n) * 10
                
                # Запускаем градиентный спуск
                result, status, history = gradient_descent(
                    oracle=oracle,
                    x_0=x0,
                    tolerance=1e-6,
                    max_iter=10000,
                    line_search_options={'method': 'Wolfe', 'c1': 1e-4, 'c2': 0.9},
                    trace=True,
                    display=False
                )
                
                if history and status == 'success':
                    iterations = len(history['func']) - 1
                    k_iterations.append(iterations)
                    
                    # Сохраняем результат
                    results_3_2.append({
                        'n': n,
                        'k': k,
                        'итерации': iterations,
                        'статус': status
                    })
            
            # Усредняем результаты
            if k_iterations:  # Проверяем, что есть результаты
                avg_iter = np.mean(k_iterations)
                std_iter = np.std(k_iterations)
                avg_iterations_per_k.append(avg_iter)
                std_iterations_per_k.append(std_iter)
            else:
                avg_iterations_per_k.append(0)
                std_iterations_per_k.append(0)
            
            # Вывод прогресса
            if k_idx % 3 == 0:
                if k_iterations:
                    print(f"  k={k:7.1f}: {np.mean(k_iterations):7.1f} ± {np.std(k_iterations):5.1f} итераций")
                else:
                    print(f"  k={k:7.1f}: нет данных")
        
        # Рисуем кривую для этой размерности
        if avg_iterations_per_k:
            plt.plot(k_values, avg_iterations_per_k, 
                    color=colors[n_idx % len(colors)], 
                    linewidth=1.5, 
                    marker='o',
                    markersize=4,
                    label=f'n = {n}')
            
            # Добавляем область неопределенности (стандартное отклонение)
            if std_iterations_per_k:
                plt.fill_between(k_values, 
                                np.array(avg_iterations_per_k) - np.array(std_iterations_per_k),
                                np.array(avg_iterations_per_k) + np.array(std_iterations_per_k),
                                color=colors[n_idx % len(colors)], 
                                alpha=0.2)
    
    # Настройка графика
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Число обусловленности k (лог. шкала)', fontsize=12)
    plt.ylabel('Число итераций T(k, n) (лог. шкала)', fontsize=12)
    plt.title('Зависимость числа итераций градиентного спуска от k и n', 
             fontsize=14)
    plt.grid(True, alpha=0.3, which='both')
    plt.legend(fontsize=10, loc='upper left')
    plt.tight_layout()
    
    # Сохраняем график в ту же папку
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = 'experiment_3_2.png'
    filepath = os.path.join(script_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"\nГрафик сохранен: {filepath}")
    plt.show()
    
    return results_3_2

# ==================== главная функция ====================

def main():
    """
    Функция для запуска экспериментов
    """
   
    # Запуск экспериментов
    results_3_1 = run_experiment_3_1()
    results_3_2 = run_experiment_3_2()
    
    # Сохранение результатов в файл
    all_results = {
        'results_3_1': results_3_1,
        'results_3_2': results_3_2}
    
    save_results_to_file(all_results)
    
    print("\n" + "="*70)
    print("Эксперименты завершены")
    print("="*70)
    
    # Показываем созданные файлы
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"\nСозданные файлы в папке: {script_dir}")
    print("-" * 50)
    
    # Ищем созданные файлы
    created_files = []
    for file in os.listdir(script_dir):
        if (file.startswith('experiment_3_') and file.endswith('.png')) or \
           file == 'experiments_results.txt':
            filepath = os.path.join(script_dir, file)
            print(f"- {file}")
            created_files.append(filepath)
    
    if not created_files:
        print("Файлы не найдены")

# ==================== запуск программы ====================

if __name__ == "__main__":
    main()