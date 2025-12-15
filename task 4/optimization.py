import numpy as np
from numpy.linalg import LinAlgError
import scipy
from scipy.optimize import line_search as wolfe_line_search
from scipy.linalg import cho_factor, cho_solve
from datetime import datetime
from collections import defaultdict


class LineSearchTool(object):
    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        else:
            raise ValueError('Unknown method {}'.format(method))

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError('LineSearchTool initializer must be of type dict')
        return cls(**options)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        """
        Finds the step size alpha for a given starting point x_k
        and for a given search direction d_k that satisfies necessary
        conditions for phi(alpha) = oracle.func(x_k + alpha * d_k).

        Parameters
        ----------
        oracle : BaseSmoothOracle-descendant object
            Oracle with .func_directional() and .grad_directional() methods implemented for computing
            function values and its directional derivatives.
        x_k : np.array
            Starting point
        d_k : np.array
            Search direction
        previous_alpha : float or None
            Starting point to use instead of self.alpha_0 to keep the progress from
            previous steps. If None, self.alpha_0, is used as a starting point.

        Returns
        -------
        alpha : float or None if failure
            Chosen step size
        """
        
        if self._method == 'Constant':
            return self.c
            
        elif self._method == 'Armijo':
            # Метод Армихо
            if previous_alpha is not None:
                alpha_0 = previous_alpha  # Используем previous_alpha как есть
            else:
                alpha_0 = self.alpha_0
                
            return self._armijo_backtracking(oracle, x_k, d_k, alpha_0)
                    
        elif self._method == 'Wolfe':
            # Метод Вульфа
            try:
                alpha, fc, gc, fval, old_fval, gval = wolfe_line_search(
                    f=oracle.func,
                    myfprime=oracle.grad,
                    xk=x_k,
                    pk=d_k,
                    c1=self.c1,
                    c2=self.c2
                )
                
                if alpha is not None:
                    return alpha
                else:
                    # Если Вульф не сработал, используем Армихо как fallback
                    return self._armijo_backtracking(oracle, x_k, d_k, self.alpha_0)
                    
            except Exception as e:
                # В случае ошибки используем Армихо
                return self._armijo_backtracking(oracle, x_k, d_k, self.alpha_0)
        
        else:
            raise ValueError('Unknown method {}'.format(self._method))

    def _armijo_backtracking(self, oracle, x_k, d_k, alpha_0):
        """
        Реализация метода бэктрекинга Армихо.
        """
        # Проверяем, что направление спуска
        grad_k = oracle.grad(x_k)
        directional_derivative = np.dot(grad_k, d_k)
        
        # Если не направление спуска или почти нулевое
        if directional_derivative >= -1e-12:
            return 0.0
        
        f_k = oracle.func(x_k)
        alpha = alpha_0
        beta = 0.5
        
        # Максимальное число итераций бэктрекинга
        max_iter = 50
        
        for i in range(max_iter):
            f_new = oracle.func(x_k + alpha * d_k)
            armijo_bound = f_k + self.c1 * alpha * directional_derivative
            
            if f_new <= armijo_bound:
                return alpha
                
            alpha *= beta
            
        # Если не нашли подходящий шаг
        return alpha


def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()


def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    
    start_time = datetime.now()
    
    # Начальные вычисления
    grad_k = oracle.grad(x_k)
    grad_norm_sq_k = np.sum(grad_k**2)
    grad_norm_sq_0 = grad_norm_sq_k
    
    # Запись начальной точки в историю
    if trace:
        history['time'].append(0.0)
        history['func'].append(oracle.func(x_k))
        history['grad_norm'].append(np.sqrt(grad_norm_sq_k))
        if x_k.size <= 2:
            history['x'].append(np.copy(x_k))
    
    # Вывод начальной точки
    if display:
        print(f"Iteration 0: f(x) = {oracle.func(x_k):.6f}, ||grad|| = {np.sqrt(grad_norm_sq_k):.6e}")
    
    # Проверка критерия остановки
    if grad_norm_sq_k <= tolerance * grad_norm_sq_0:
        return x_k, 'success', history
    
    previous_alpha = None
    
    for k in range(1, max_iter + 1):
        # Направление спуска
        d_k = -grad_k
        
        # Линейный поиск
        alpha = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha)
        
        if alpha is None or np.isnan(alpha) or np.isinf(alpha) or alpha <= 0:
            return x_k, 'computational_error', history
        
        # Обновление точки
        x_k = x_k + alpha * d_k
        
        # Вычисление градиента в новой точке
        grad_k = oracle.grad(x_k)
        grad_norm_sq_k = np.sum(grad_k**2)
        
        # Запись в историю
        if trace:
            current_time = (datetime.now() - start_time).total_seconds()
            history['time'].append(current_time)
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(np.sqrt(grad_norm_sq_k))
            if x_k.size <= 2:
                history['x'].append(np.copy(x_k))
        
        # Вывод информации
        if display:
            print(f"Iteration {k}: f(x) = {oracle.func(x_k):.6f}, ||grad|| = {np.sqrt(grad_norm_sq_k):.6e}, alpha = {alpha:.6f}")
        
        # Проверка критерия остановки
        if grad_norm_sq_k <= tolerance * grad_norm_sq_0:
            return x_k, 'success', history
        
        previous_alpha = alpha
    
    return x_k, 'iterations_exceeded', history


def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    
    start_time = datetime.now()
    
    # Начальные вычисления
    grad_k = oracle.grad(x_k)
    
    # Проверка на численные ошибки
    if np.any(np.isnan(grad_k)) or np.any(np.isinf(grad_k)):
        return x_k, 'computational_error', history
    
    grad_norm_sq_k = np.sum(grad_k**2)
    grad_norm_sq_0 = grad_norm_sq_k
    
    # Запись начальной точки в историю
    if trace:
        history['time'].append(0.0)
        func_val = oracle.func(x_k)
        if np.isnan(func_val) or np.isinf(func_val):
            return x_k, 'computational_error', history
        history['func'].append(func_val)
        history['grad_norm'].append(np.sqrt(grad_norm_sq_k))
        if x_k.size <= 2:
            history['x'].append(np.copy(x_k))
    
    # Вывод начальной точки
    if display:
        print(f"Iteration 0: f(x) = {oracle.func(x_k):.6f}, ||grad|| = {np.sqrt(grad_norm_sq_k):.6e}")
    
    # Проверка критерия остановки
    if grad_norm_sq_k <= tolerance * grad_norm_sq_0:
        return x_k, 'success', history
    
    previous_alpha = None
    
    for k in range(1, max_iter + 1):
        # Вычисляем гессиан и решаем систему для направления Ньютона
        try:
            hess_k = oracle.hess(x_k)
            
            # Проверка гессиана на численные ошибки
            if np.any(np.isnan(hess_k)) or np.any(np.isinf(hess_k)):
                return x_k, 'computational_error', history
            
            # Для одномерного случая: если гессиан близок к нулю, а градиент не ноль, то computational_error
            if hess_k.size == 1 and np.abs(hess_k) < 1e-15 and np.abs(grad_k) > 1e-15:
                return x_k, 'computational_error', history
            
            # Используем разложение Холецкого (как требуется в задании)
            c, lower = cho_factor(hess_k)
            d_k = cho_solve((c, lower), -grad_k)
                
        except (LinAlgError, ValueError) as e:
            # Ошибка при вычислении направления Ньютона
            if display:
                print(f"Newton direction error at iteration {k}: {e}")
            return x_k, 'newton_direction_error', history
        
        # Проверка направления на численные ошибки
        if np.any(np.isnan(d_k)) or np.any(np.isinf(d_k)):
            return x_k, 'computational_error', history
        
        # Линейный поиск
        alpha = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha)
        
        # Проверка на ошибки
        if alpha is None or np.isnan(alpha) or np.isinf(alpha) or alpha <= 0:
            return x_k, 'computational_error', history
        
        # Обновление точки
        x_k = x_k + alpha * d_k
        
        # Проверка на переполнение
        if np.any(np.abs(x_k) > 1e100):
            return x_k, 'computational_error', history
        
        # Проверка новой точки на численные ошибки
        if np.any(np.isnan(x_k)) or np.any(np.isinf(x_k)):
            return x_k, 'computational_error', history
        
        # Вычисление градиента в новой точке
        grad_k = oracle.grad(x_k)
        
        # Проверка градиента на численные ошибки
        if np.any(np.isnan(grad_k)) or np.any(np.isinf(grad_k)):
            return x_k, 'computational_error', history
            
        grad_norm_sq_k = np.sum(grad_k**2)
        
        # Проверка функции на численные ошибки
        func_val = oracle.func(x_k)
        if np.isnan(func_val) or np.isinf(func_val):
            return x_k, 'computational_error', history
        
        # Сохранение в историю
        if trace:
            current_time = (datetime.now() - start_time).total_seconds()
            history['time'].append(current_time)
            history['func'].append(func_val)
            history['grad_norm'].append(np.sqrt(grad_norm_sq_k))
            if x_k.size <= 2:
                history['x'].append(np.copy(x_k))
        
        # Отображение информации
        if display:
            print(f"Iteration {k}: f(x) = {oracle.func(x_k):.6f}, "
                  f"||grad|| = {np.sqrt(grad_norm_sq_k):.6e}, alpha = {alpha:.6f}")
        
        # Проверка критерия остановки
        if grad_norm_sq_k <= tolerance * grad_norm_sq_0:
            return x_k, 'success', history
        
        previous_alpha = alpha
    
    # Если превышено максимальное число итераций
    return x_k, 'iterations_exceeded', history