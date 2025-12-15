import numpy as np
import scipy
from scipy.special import expit
import scipy.sparse


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """
    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')
    
    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')
    
    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A 


class LogRegL2Oracle(BaseSmoothOracle):
    """
    Oracle for logistic regression with l2 regularization:
         func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef
        self.m = len(b)
        
    def func(self, x):
        # Вычисляем Ax
        Ax = self.matvec_Ax(x)
        
        # Вычисляем -b_i * a_i^T x = -b * (Ax) поэлементно
        z = -self.b * Ax
        
        # Используем logaddexp для устойчивого вычисления log(1 + exp(z))
        # log(1 + exp(z)) = logaddexp(0, z)
        log_terms = np.logaddexp(0, z)
        
        # Суммируем и делим на m
        data_term = np.sum(log_terms) / self.m
        
        # Добавляем регуляризацию
        reg_term = 0.5 * self.regcoef * np.dot(x, x)
        
        return data_term + reg_term

    def grad(self, x):
        # Вычисляем Ax
        Ax = self.matvec_Ax(x)
        
        # Вычисляем сигмоиду: σ(-b_i * a_i^T x) = 1/(1 + exp(b_i * a_i^T x))
        # Но expit(z) = 1/(1 + exp(-z)), поэтому:
        # 1/(1 + exp(b_i * a_i^T x)) = expit(-b_i * a_i^T x)
        z = -self.b * Ax
        sigma = expit(z)  # σ(-b_i * a_i^T x)
        
        # Вычисляем градиент: 1/m * A^T * (-b * σ(-b_i * a_i^T x)) + regcoef * x
        # где -b * σ(-b_i * a_i^T x) = -b * sigma
        grad_data = self.matvec_ATx(-self.b * sigma) / self.m
        
        # Добавляем регуляризацию
        grad_reg = self.regcoef * x
        
        return grad_data + grad_reg

    def hess(self, x):
        # Вычисляем Ax
        Ax = self.matvec_Ax(x)
        
        # Вычисляем сигмоиду
        z = -self.b * Ax
        sigma = expit(z)  # σ(-b_i * a_i^T x)
        
        # Вычисляем диагональную матрицу S: s_i = σ(z_i) * (1 - σ(z_i))
        # где σ(z) = expit(z)
        s = sigma * (1 - sigma)
        
        # Вычисляем гессиан: 1/m * A^T * Diag(s) * A + regcoef * I
        hess_data = self.matmat_ATsA(s) / self.m
        
        # Добавляем регуляризацию
        n = x.shape[0]
        if scipy.sparse.issparse(hess_data):
            hess_reg = self.regcoef * scipy.sparse.eye(n, format=hess_data.format)
            hess_result = hess_data + hess_reg
            # Для совместимости с тестами, преобразуем в плотную матрицу
            return hess_result.toarray()
        else:
            hess_reg = self.regcoef * np.eye(n)
            return hess_data + hess_reg

    def func_directional(self, x, d, alpha):
        return super().func_directional(x, d, alpha)

    def grad_directional(self, x, d, alpha):
        return super().grad_directional(x, d, alpha)


class LogRegL2OptimizedOracle(LogRegL2Oracle):
    """
    Oracle for logistic regression with l2 regularization
    with optimized *_directional methods (are used in line_search).
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)
        self._last_x = None
        self._last_Ax = None
        self._last_d = None
        self._last_Ad = None
        self._last_test_point = None  # x̂ = x + αd
        self._last_test_Ax = None     # A x̂
        
    def _compute_Ax_if_needed(self, x):
        """Вычисляет Ax, если x изменился с последнего раза."""
        if self._last_x is None or not np.array_equal(x, self._last_x):
            # Проверим, может быть x совпадает с последней тестовой точкой
            if self._last_test_point is not None and np.array_equal(x, self._last_test_point):
                self._last_x = x.copy()
                self._last_Ax = self._last_test_Ax.copy()
            else:
                self._last_x = x.copy()
                self._last_Ax = self.matvec_Ax(x)
        return self._last_Ax
    
    def _compute_Ad_if_needed(self, d):
        """Вычисляет Ad, если d изменился с последнего раза."""
        if self._last_d is None or not np.array_equal(d, self._last_d):
            self._last_d = d.copy()
            self._last_Ad = self.matvec_Ax(d)
        return self._last_Ad
    
    def func(self, x):
        # Проверим, может быть x совпадает с последней тестовой точкой
        if self._last_test_point is not None and np.array_equal(x, self._last_test_point):
            Ax = self._last_test_Ax
            self._last_x = x.copy()
            self._last_Ax = Ax
        else:
            Ax = self._compute_Ax_if_needed(x)
        
        z = -self.b * Ax
        log_terms = np.logaddexp(0, z)
        data_term = np.sum(log_terms) / self.m
        reg_term = 0.5 * self.regcoef * np.dot(x, x)
        
        return data_term + reg_term
    
    def grad(self, x):
        # Используем кэширование
        if self._last_test_point is not None and np.array_equal(x, self._last_test_point):
            Ax = self._last_test_Ax
            self._last_x = x.copy()
            self._last_Ax = Ax
        else:
            Ax = self._compute_Ax_if_needed(x)
        
        z = -self.b * Ax
        sigma = expit(z)
        grad_data = self.matvec_ATx(-self.b * sigma) / self.m
        grad_reg = self.regcoef * x
        
        return grad_data + grad_reg
    
    def hess(self, x):
        # При вычислении гессиана используем кэш, но не сбрасываем его
        if self._last_test_point is not None and np.array_equal(x, self._last_test_point):
            Ax = self._last_test_Ax
            self._last_x = x.copy()
            self._last_Ax = Ax
        else:
            Ax = self._compute_Ax_if_needed(x)
        
        # Вычисляем сигмоиду
        z = -self.b * Ax
        sigma = expit(z)
        s = sigma * (1 - sigma)
        
        # Вычисляем гессиан
        hess_data = self.matmat_ATsA(s) / self.m
        
        # Добавляем регуляризацию
        n = x.shape[0]
        if scipy.sparse.issparse(hess_data):
            hess_reg = self.regcoef * scipy.sparse.eye(n, format=hess_data.format)
            hess_result = hess_data + hess_reg
            return hess_result.toarray()
        else:
            hess_reg = self.regcoef * np.eye(n)
            return hess_data + hess_reg
    
    def func_directional(self, x, d, alpha):
        # Вычисляем точку x̂ = x + αd
        x_hat = x + alpha * d
        
        # Проверяем, не вычисляли ли мы уже A x̂
        if self._last_test_point is not None and np.array_equal(x_hat, self._last_test_point):
            A_x_hat = self._last_test_Ax
        else:
            # Вычисляем Ax и Ad
            Ax = self._compute_Ax_if_needed(x)
            Ad = self._compute_Ad_if_needed(d)
            
            # Вычисляем A(x + αd) = Ax + α*Ad
            A_x_hat = Ax + alpha * Ad
            
            # Сохраняем в кэш тестовой точки
            self._last_test_point = x_hat.copy()
            self._last_test_Ax = A_x_hat.copy()
        
        # Вычисляем функцию в точке x̂
        z = -self.b * A_x_hat
        log_terms = np.logaddexp(0, z)
        data_term = np.sum(log_terms) / self.m
        reg_term = 0.5 * self.regcoef * np.dot(x_hat, x_hat)
        
        return data_term + reg_term
    
    def grad_directional(self, x, d, alpha):
        # Вычисляем точку x̂ = x + αd
        x_hat = x + alpha * d
        
        # Всегда вычисляем Ad (используем кэш для d)
        Ad = self._compute_Ad_if_needed(d)
        
        # Проверяем, не вычисляли ли мы уже A x̂
        if self._last_test_point is not None and np.array_equal(x_hat, self._last_test_point):
            A_x_hat = self._last_test_Ax
        else:
            # Вычисляем Ax
            Ax = self._compute_Ax_if_needed(x)
            
            # Вычисляем A(x + αd) = Ax + α*Ad
            A_x_hat = Ax + alpha * Ad
            
            # Сохраняем в кэш тестовой точки
            self._last_test_point = x_hat.copy()
            self._last_test_Ax = A_x_hat.copy()
        
        # Вычисляем производную по направлению
        z = -self.b * A_x_hat
        sigma = expit(z)
        
        # φ'(α) = d^T ∇f(x+αd) = (1/m) * (Ad)^T * (-b * σ(-b * A(x+αd))) + regcoef * d^T (x+αd)
        directional_derivative = (1.0 / self.m) * np.dot(Ad, -self.b * sigma) + self.regcoef * np.dot(d, x_hat)
        
        return directional_derivative
    
    def clear_cache(self):
        """Очистка кэша."""
        self._last_x = None
        self._last_Ax = None
        self._last_d = None
        self._last_Ad = None
        self._last_test_point = None
        self._last_test_Ax = None


def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    """
    Auxiliary function for creating logistic regression oracles.
    """
    # Определяем функции для работы с матрицами
    if scipy.sparse.issparse(A):
        # Для разреженных матриц
        matvec_Ax = lambda x: A.dot(x)
        matvec_ATx = lambda x: A.T.dot(x)
        
        def matmat_ATsA(s):
            # A^T * Diag(s) * A
            # Для эффективности: (A.T * s) @ A, но для разреженных матриц
            if scipy.sparse.issparse(A):
                # Преобразуем s в диагональную матрицу
                s_diag = scipy.sparse.diags(s)
                result = A.T @ (s_diag @ A)
                # Возвращаем разреженную матрицу
                return result
            else:
                return A.T @ (s[:, np.newaxis] * A)
    else:
        # Для плотных матриц
        matvec_Ax = lambda x: A.dot(x)
        matvec_ATx = lambda x: A.T.dot(x)
        
        def matmat_ATsA(s):
            # A^T * Diag(s) * A = (A.T * s) @ A
            # Используем поэлементное умножение и матричное умножение
            return (A.T * s) @ A
    
    if oracle_type == 'usual':
        oracle = LogRegL2Oracle
    elif oracle_type == 'optimized':
        oracle = LogRegL2OptimizedOracle
    else:
        raise ValueError('Unknown oracle_type=%s' % oracle_type)
    
    return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)


def grad_finite_diff(func, x, eps=1e-8):
    """
    Returns approximation of the gradient using finite differences.
    """
    n = len(x)
    grad = np.zeros(n)
    f_x = func(x)
    
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1
        f_x_eps = func(x + eps * e_i)
        grad[i] = (f_x_eps - f_x) / eps
    
    return grad


def hess_finite_diff(func, x, eps=1e-5):
    """
    Returns approximation of the Hessian using finite differences.
    """
    n = len(x)
    hess = np.zeros((n, n))
    f_x = func(x)
    
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1
        f_x_eps_i = func(x + eps * e_i)
        
        for j in range(n):
            e_j = np.zeros(n)
            e_j[j] = 1
            f_x_eps_j = func(x + eps * e_j)
            f_x_eps_ij = func(x + eps * e_i + eps * e_j)
            
            hess[i, j] = (f_x_eps_ij - f_x_eps_i - f_x_eps_j + f_x) / (eps * eps)
    
    return hess