"""
Эксперимент 3.3:
   Тестирование методов градиентного спуска и Ньютона на реальных данных из LIBSVM
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import time
import warnings
from sklearn.datasets import load_svmlight_file

# Импорт реализаций
from oracles import create_log_reg_oracle
from optimization import gradient_descent, newton

warnings.filterwarnings('ignore')

def load_libsvm_data(dataset_name, data_dir="./data"):
    """
    Загрузка данных из LIBSVM
    """
    file_paths = {
        'w8a': f'{data_dir}/w8a',
        'gisette': f'{data_dir}/gisette_scale',
        'real-sim': f'{data_dir}/real-sim'}
    
    dataset_path = file_paths[dataset_name]
    
    try:
        # Загрузка данных
        A, b = load_svmlight_file(dataset_path)
        
        # Преобразование меток в {-1, 1}
        if set(b) == {0, 1}:
            b = 2 * b - 1
        elif set(b) == {-1, 1}:
            pass  # Уже в нужном формате
        else:
            # Для многоклассовых датасетов берём только два класса
            unique_classes = np.unique(b)
            if len(unique_classes) > 2:
                # Бинаризация: первый класс против остальных
                b = np.where(b == unique_classes[0], 1, -1)
            
        print(f"Датасет '{dataset_name}' загружен:")
        print(f"  Образцы: {A.shape[0]}, Признаки: {A.shape[1]}")
        print(f"  Классы: {set(b)}")
        
        return A, b
        
    except FileNotFoundError:
        print(f"Файл {dataset_path} не найден.")
        print("Скачайте датасеты и поместите в директорию data/")
        return None, None

def run_optimization_experiment(dataset_name, method='both'):
    """
    Запуск оптимизации
    """
    print(f"\n{'='*60}")
    print(f"Эксперимент 3.3: {dataset_name}")
    print('='*60)
    
    # Загрузка данных
    A, b = load_libsvm_data(dataset_name)
    if A is None:
        return None
    
    m, n = A.shape
    print(f"\nСтатистика датасета:")
    print(f"  m (образцы) = {m:,}")
    print(f"  n (признаки) = {n:,}")
    
    # Коэффициент регуляризации λ = 1/m
    regcoef = 1.0 / m
    print(f"  λ (regcoef) = {regcoef:.6f}")
    
    # Создание оракула
    oracle = create_log_reg_oracle(A, b, regcoef, oracle_type='usual')
    
    # Начальная точка x0 = 0
    x0 = np.zeros(n)
    
    # Параметры методов
    # Относительная точность ε = 1e-5
    epsilon = 1e-5
    max_iter_gd = 10000
    max_iter_newton = 100
    
    # Параметры линейного поиска по умолчанию (условия Вульфа)
    line_search_options = {'method': 'Wolfe'}
    
    results = {}
    
    # Вычисляем начальную норму градиента для относительного критерия
    grad_x0 = oracle.grad(x0)
    grad_norm_x0 = np.linalg.norm(grad_x0)
    print(f"\nНачальная норма градиента: ||∇f(x0)|| = {grad_norm_x0:.6e}")
    print(f"Критерий остановки: ||∇f(xk)|| ≤ {epsilon} * {grad_norm_x0:.6e} = {epsilon * grad_norm_x0:.6e}")
    
    # Функция для проверки критерия остановки
    def check_stopping_criterion(grad_norm):
        return grad_norm <= epsilon * grad_norm_x0
    
    # Запуск градиентного спуска
    if method in ['gd', 'both']:
        print(f"\nГрадиентный спуск")
        print("-" * 40)
        
        gd_start = time.time()
        
        # Запускаем градиентный спуск
        x_gd, status_gd, history_gd = gradient_descent(
            oracle=oracle,
            x_0=x0,
            tolerance=epsilon,  # Относительная точность
            max_iter=max_iter_gd,
            line_search_options=line_search_options,
            trace=True,
            display=False)
        
        gd_time = time.time() - gd_start
        
        # Обрабатываем историю
        if history_gd:
            # Проверяем критерий остановки для каждой итерации
            for i in range(len(history_gd['grad_norm'])):
                if check_stopping_criterion(history_gd['grad_norm'][i]):
                    # Обрезаем историю до момента достижения критерия
                    for key in history_gd:
                        history_gd[key] = history_gd[key][:i+1]
                    break
        
        print(f"  Статус: {status_gd}")
        print(f"  Время: {gd_time:.2f} секунд")
        print(f"  Итераций: {len(history_gd['func']) if history_gd else 0}")
        
        if history_gd and len(history_gd['func']) > 0:
            final_func = history_gd['func'][-1]
            final_grad_norm = history_gd['grad_norm'][-1]
            print(f"  Финальное f(x): {final_func:.6e}")
            print(f"  Финальная ||∇f(x)||: {final_grad_norm:.6e}")
            print(f"  Отношение: ||∇f(x)||/||∇f(x0)|| = {final_grad_norm/grad_norm_x0:.6e}")
        
        results['gd'] = {
            'x': x_gd,
            'status': status_gd,
            'history': history_gd,
            'time': gd_time,
            'iterations': len(history_gd['func']) if history_gd else 0}
    
    # Запуск метода Ньютона
    if method in ['newton', 'both']:
        print(f"\nМетод Ньютона")
        print("-" * 40)
        
        newton_start = time.time()
        
        # Запускаем метод Ньютона
        x_newton, status_newton, history_newton = newton(
            oracle=oracle,
            x_0=x0,
            tolerance=epsilon,  # Относительная точность
            max_iter=max_iter_newton,
            line_search_options=line_search_options,
            trace=True,
            display=False)
        
        newton_time = time.time() - newton_start
        
        # Обрабатываем историю
        if history_newton:
            # Проверяем критерий остановки для каждой итерации
            for i in range(len(history_newton['grad_norm'])):
                if check_stopping_criterion(history_newton['grad_norm'][i]):
                    # Обрезаем историю до момента достижения критерия
                    for key in history_newton:
                        history_newton[key] = history_newton[key][:i+1]
                    break
        
        print(f"  Статус: {status_newton}")
        print(f"  Время: {newton_time:.2f} секунд")
        print(f"  Итераций: {len(history_newton['func']) if history_newton else 0}")
        
        if history_newton and len(history_newton['func']) > 0:
            final_func = history_newton['func'][-1]
            final_grad_norm = history_newton['grad_norm'][-1]
            print(f"  Финальное f(x): {final_func:.6e}")
            print(f"  Финальная ||∇f(x)||: {final_grad_norm:.6e}")
            print(f"  Отношение: ||∇f(x)||/||∇f(x0)|| = {final_grad_norm/grad_norm_x0:.6e}")
        
        results['newton'] = {
            'x': x_newton,
            'status': status_newton,
            'history': history_newton,
            'time': newton_time,
            'iterations': len(history_newton['func']) if history_newton else 0}
    
    return results

def plot_convergence_graphs(results, dataset_name, save_dir="plots"):
    """
    Построение графиков сходимости
    """
    if not results or ('gd' not in results and 'newton' not in results):
        print(f"Нет данных для построения графиков ({dataset_name})")
        return
    
    # Создаем директорию для графиков
    os.makedirs(save_dir, exist_ok=True)
    
    # Настройки стиля
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = [12, 5]
    plt.rcParams['font.size'] = 12
    plt.rcParams['lines.linewidth'] = 2
    
    colors = {'gd': '#1f77b4', 'newton': '#d62728'}
    labels = {'gd': 'Градиентный спуск', 'newton': 'Метод Ньютона'}
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # График (a): Значение функции от реального времени работы метода
    for method in ['gd', 'newton']:
        if method in results and results[method]['history']:
            hist = results[method]['history']
            if 'time' in hist and len(hist['time']) > 0:
                times = hist['time']
                func_vals = hist['func']
                
                # Начинаем время с 0
                if len(times) > 0:
                    start_time = times[0]
                    times = [t - start_time for t in times]
                
                ax1.plot(times, func_vals,
                        color=colors[method],
                        label=labels[method],
                        marker='o' if len(times) < 20 else None,
                        markersize=4,
                        markevery=max(1, len(times)//10))
    
    ax1.set_xlabel('Время работы (секунды)', fontsize=12)
    ax1.set_ylabel('Значение функции f(x)', fontsize=12)
    ax1.set_title(f'(a) f(x) от времени\nДатасет: {dataset_name}', fontsize=13)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=11)
    
    # График (b): Относительный квадрат нормы градиента от времени (логарифмическая шкала)
    for method in ['gd', 'newton']:
        if method in results and results[method]['history']:
            hist = results[method]['history']
            if 'time' in hist and 'grad_norm' in hist and len(hist['time']) > 0:
                times = hist['time']
                grad_norms = hist['grad_norm']
                
                if len(grad_norms) > 0:
                    # Вычисляем относительный квадрат нормы градиента
                    # Находим начальную норму градиента (первая итерация)
                    grad_norm_0 = grad_norms[0] if len(grad_norms) > 0 else 1.0
                    relative_grad_norm_sq = [(gn**2) / (grad_norm_0**2) for gn in grad_norms]
                    
                    # Начинаем время с 0
                    if len(times) > 0:
                        start_time = times[0]
                        times = [t - start_time for t in times]
                    
                    ax2.plot(times, relative_grad_norm_sq,
                            color=colors[method],
                            label=labels[method],
                            marker='o' if len(times) < 20 else None,
                            markersize=4,
                            markevery=max(1, len(times)//10))
    
    ax2.set_xlabel('Время работы (секунды)', fontsize=12)
    ax2.set_ylabel(r'$\|\nabla f(x_k)\|^2 / \|\nabla f(x_0)\|^2$', fontsize=12)
    ax2.set_title(f'(b) Относительная норма градиента\nДатасет: {dataset_name}', fontsize=13)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, linestyle='--', which='both')
    ax2.legend(fontsize=11)
    
    plt.suptitle(f'Эксперимент 3.3: Сравнение методов оптимизации\nДатасет: {dataset_name}', 
                 fontsize=14, y=1.02)
    
    plt.tight_layout()
    
    # Сохраняем график
    filename = f'{save_dir}/convergence_{dataset_name}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  График сохранен: {filename}")
    
    plt.show()
    plt.close()

def analyze_computational_complexity(results_dict, datasets_info):
    """
    Анализ вычислительной сложности методов
    """
    analysis = []
    analysis.append("\n" + "="*60)
    analysis.append("Анализ вычислительной сложности")
    analysis.append("="*60)
    
    analysis.append("\nТеоретическая сложность на итерацию:")
    analysis.append("  Градиентный спуск:")
    analysis.append("    - Вычисление градиента: O(m·n)")
    analysis.append("    - Память: O(m + n)")
    analysis.append("    - Общая: O(m·n)")
    
    analysis.append("\n  Метод Ньютона:")
    analysis.append("    - Вычисление градиента: O(m·n)")
    analysis.append("    - Вычисление гессиана: O(m·n²)")
    analysis.append("    - Решение системы (разложение Холецкого): O(n³)")
    analysis.append("    - Память: O(m + n²)")
    analysis.append("    - Общая: O(m·n² + n³)")
    
    analysis.append("\n" + "="*60)
    analysis.append("Экспериментальные результаты")
    analysis.append("="*60)
    
    # Таблица с результатами
    analysis.append("\n{:20} | {:20} | {:10} | {:10} | {:12} | {:10}".format(
        "Датасет", "Метод", "Итераций", "Время (с)", "Время/итер (с)", "‖∇f(x)‖/‖∇f(x₀)‖"))
    analysis.append("-" * 100)
    
    for dataset_name, results in results_dict.items():
        if dataset_name in datasets_info:
            m = datasets_info[dataset_name]['m']
            n = datasets_info[dataset_name]['n']
            
            for method in ['gd', 'newton']:
                if method in results:
                    res = results[method]
                    method_name = "Градиентный спуск" if method == 'gd' else "Метод Ньютона"
                    
                    iterations = res['iterations']
                    total_time = res['time']
                    time_per_iter = total_time / iterations if iterations > 0 else 0
                    
                    # Вычисляем отношение норм градиента
                    if res['history'] and len(res['history']['grad_norm']) > 0:
                        grad_norm_final = res['history']['grad_norm'][-1]
                        grad_norm_initial = res['history']['grad_norm'][0]
                        grad_ratio = grad_norm_final / grad_norm_initial
                    else:
                        grad_ratio = 0
                    
                    analysis.append("{:20} | {:15} | {:10} | {:10.2f} | {:12.6f} | {:10.2e}".format(
                        dataset_name, method_name, iterations, total_time, time_per_iter, grad_ratio))
    
    return "\n".join(analysis)

def save_results_to_file(results_dict, datasets_info, filename="experiment_3_3_results.txt"):
    """
    Сохранение результатов в текстовый файл.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("Эксперимент 3.3: Сравнение методов градиентного спуска и Ньютона\n")
        
        # Сохраняем анализ сложности
        complexity_analysis = analyze_computational_complexity(results_dict, datasets_info)
        f.write(complexity_analysis)
        
        f.write("\n\n" + "="*70)
        f.write("\nВsdjls:\n")
        f.write("="*70 + "\n\n")
        
        f.write("1. Градиентный спуск:\n")
        f.write("   - Преимущество: Низкие требования к памяти (O(m+n))\n")
        f.write("   - Недостаток: Медленная сходимость (линейная)\n")
        f.write("   - Применение: Большие датасеты с высокой размерностью\n\n")
        
        f.write("2. Метод Ньютона:\n")
        f.write("   - Преимущество: Быстрая сходимость (квадратичная)\n")
        f.write("   - Недостаток: Высокие требования к памяти (O(n²)) и времени (O(n³))\n")
        f.write("   - Применение: Задачи малой и средней размерности\n\n")
    
    print(f"\nРезультаты сохранены в файл: {filename}")

def main():
    """
    Основная функция для проведения эксперимента 3.3.
    """
    print("="*60)
    print("Эксперимент 3.3: Сравнение методов на реальных данных")
    print("="*60)    
    
    # Фиксация seed для воспроизводимости
    np.random.seed(23)
    
    # Список датасетов для тестирования
    datasets = ['w8a', 'gisette', 'real-sim']
    
    # Информация о датасетах (m, n)
    datasets_info = {
        'w8a': {'m': 49749, 'n': 300},
        'gisette': {'m': 6000, 'n': 5000},
        'real-sim': {'m': 72309, 'n': 20958}
    }
    
    print("\nДатасеты для тестирования:")
    for i, ds in enumerate(datasets, 1):
        info = datasets_info.get(ds, {})
        print(f"  {i}. {ds:10} (m={info.get('m', '?'):,}, n={info.get('n', '?'):,})")
    
    
    # Запускаем эксперименты на всех датасетах
    all_results = {}
    
    for dataset in datasets:
        try:
                        
            # Запускаем оптимизацию
            results = run_optimization_experiment(dataset, method='both')
            
            if results:
                all_results[dataset] = results
                
                # Строим графики сходимости
                plot_convergence_graphs(results, dataset, save_dir="convergence_plots")
                
            else:
                print(f"Не удалось получить результаты для {dataset}")
                
        except Exception as e:
            print(f"\nОшибка при обработке {dataset}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Сохраняем результаты
    if all_results:
        save_results_to_file(all_results, datasets_info, 
                           f"experiment_3_3_results.txt")
        
        # Выводим итоговую сводку
        print("Эксперимент 3.3 завершен")
        print("="*70)
        
        print(analyze_computational_complexity(all_results, datasets_info))
        
        print("\n" + "="*70)
        print("Выводы:")
        print("="*70)
        print("1. Градиентный спуск лучше подходит для больших датасетов")
        print("2. Метод Ньютона сходится быстрее, но требует больше памяти")
        print("3. Выбор метода зависит от размерности задачи и доступных ресурсов")
    else:
        print("\nНе удалось получить результаты ни для одного датасета.")

if __name__ == "__main__":
    main()