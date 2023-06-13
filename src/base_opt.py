from typing import List, Optional
from autograd import grad, jacobian, elementwise_grad
import numpy as np
import matplotlib.pyplot as plt
import sys, os
from tqdm import tqdm
import math
import scipy 
from scipy import linalg
from numpy import linalg as LA
import sys

sys.path.append("../")

import numpy as np
from datetime import datetime
from typing import Optional
from .problems import BaseSaddle

def metrics1(problem, x, y):
    """
        The function has been taken from 
        https://github.com/tkkiran/LiftedPrimalDual/blob/20696d51113551d0fe965d0f15ba904bb01f65c0/minmax/probs.py#L22
    """
    metrics_dict = {}
    z = (x, y)

    if problem.primal_func is not None and problem.dual_func is not None:
        p_func = problem.primal_func(x, y=y)
        d_func = problem.dual_func(y, x=x)
        gap = p_func - d_func
        try:
            assert gap >= 0 or np.isclose(gap, 0.)
        except:
            print('p_func={}, d_func={},gap={}'.format(p_func, d_func, gap))
            raise ValueError('Gap is negative!')
        metrics_dict['gap'] = gap
    
    grad_x, grad_y = problem.grad(x, y)
    if problem._proj_x is not None:
        x = z[0]
        stepsize_x = 1/problem.L/10
        grad_x = (x - problem.proj_x(x - stepsize_x*grad_x))/stepsize_x
    if problem._proj_y is not None:
        y = z[1]
        stepsize_y = 1/problem.L/10
        grad_y = (y - problem.proj_y(y - stepsize_y*grad_y))/stepsize_y

    metrics_dict['grad_norm'] = ((grad_x**2).sum() + (grad_y**2).sum())**0.5
    metrics_dict['func'] = problem.F(x, y)

    metrics_string = ""
    if problem.primal_func is not None and problem.dual_func is not None:
        metrics_string += f"gap={metrics_dict['gap']:2.2g}"
    metrics_string += ",|grad|={metrics_dict['grad_norm']:2.2g},func={metrics_dict['func']:.2g}"
    return metrics_dict, metrics_string


class BaseSaddleOpt(object):
    """
    Base class for saddle-point algorithms.
    Parameters
    ----------
    oracle: BaseSmoothSaddleOracle
        Oracle corresponding to the objective function.
    z_0: ArrayPair
        Initial guess
    tolerance: Optional[float]
        Accuracy required for stopping criteria.
    stopping_criteria: Optional[str]
        Str specifying stopping criteria. Supported values:
        "grad_rel": terminate if ||f'(x_k)||^2 / ||f'(x_0)||^2 <= eps
        "grad_abs": terminate if ||f'(x_k)||^2 <= eps
    logger: Optional[Logger]
        Stores the history of the method during its iterations.
    """
    def __init__(
        self,
        problem: BaseSaddle,
        x0: np.ndarray, 
        y0: np.ndarray,
        eps: float,
        stopping_criteria: Optional[str],
        params: dict
    ):
        self.problem = problem
        self.x = x0.copy()
        self.y = y0.copy()

        if hasattr(self.problem, 'proj_x'):
            self.proj_x = self.problem.proj_x
        else:
            self.proj_x = lambda x: x

        if hasattr(self.problem, 'proj_y'):
            self.proj_y = self.problem.proj_y
        else:
            self.proj_y = lambda y: y

        self.x = self.proj_x(self.x)
        self.y = self.proj_y(self.y)

        self.params = params
        
        self.eps = eps
        
        # Amount of atomic operations: 
        # +, -, *, /, sqrt, **, dot = 1
        # A.dot(x) = N, where A - N x m matrix
        # A.dot(B) = N*M, where B - m x M matrix
        self._step_complexity = None
        
        if stopping_criteria == 'grad_rel':
            self.stopping_criteria = self.stopping_criteria_grad_relative
        elif stopping_criteria == 'grad_abs':
            self.stopping_criteria = self.stopping_criteria_grad_absolute
        elif stopping_criteria == 'loss':
            self.stopping_criteria = self.stopping_criteria_loss
        elif stopping_criteria == 'sliding':
            self.stopping_criteria = self.stopping_criteria_sliding
        elif stopping_criteria == 'accel_sliding':
            self.stopping_criteria = self.stopping_criteria_acceg_sliding
        elif stopping_criteria == None:
            self.stopping_criteria = self.stopping_criteria_none
        elif callable(stopping_criteria):
            self.stopping_criteria = stopping_criteria
        else:
            raise ValueError('Unknown stopping criteria type: "{}"' \
                             .format(stopping_criteria))

    def __call__(self, 
            max_iter: int, 
            max_time: float = None,
            verbose=1
           ):
        """
        Run the method for no more that max_iter iterations and max_time seconds.
        Parameters
        ----------
        max_iter: int
            Maximum number of iterations.
        max_time: float
            Maximum time (in seconds).
        """
        self.grad_norm_0 = np.sqrt(self.x.dot(self.x)**2 + self.y.dot(self.y)**2)
   
        if max_time is None:
            max_time = +np.inf
        if not hasattr(self, 'time'):
            self.time = 0.
        
        self.loss = [self.problem.loss(self.x, self.y)]
        self.all_metrics = {"gap": [],
                            "grad_norm": [],
                            "func": [],
                            "step_time": [],
                            "step_complexity": []
                           }
        metrics, _ = metrics1(self.problem, self.x, self.y)
        for metric, val in metrics.items():
            self.all_metrics[metric].append(val)
        self.x_hist, self.y_hist = [self.x], [self.y]

        bar = range(max_iter)
        if verbose > 0:
            bar = tqdm(bar, desc=self.__class__.__name__)
            
        self._absolute_time = datetime.now()
        
        for iter_count in bar:
            self.iter_count = iter_count
            if self.time > max_time:
                break
            self._update_time()
            
            self.step()
            self.all_metrics["step_time"].append((datetime.now() -
                                                  self._absolute_time).total_seconds())
            
            self.all_metrics["step_complexity"].append(self._step_complexity)
            self.x = self.proj_x(self.x)
            self.y = self.proj_y(self.y)
            lo = self.problem.loss(self.x, self.y)
            metrics, _ = metrics1(self.problem, self.x, self.y)
            for metric, val in metrics.items():
                self.all_metrics[metric].append(val)
            
            self.loss.append(lo)  
            self.x_hist.append(self.x)
            self.y_hist.append(self.y)            
            
            if self.stopping_criteria():
                break
        return self.loss, self.x_hist, self.y_hist
                

    def _update_time(self):
        now = datetime.now()
        self.time += (now - self._absolute_time).total_seconds()
        self._absolute_time = now

    def step(self):
        raise NotImplementedError('step() not implemented!')

    def stopping_criteria_grad_relative(self):
        eta_x, eta_y = self.problem._optimiser_params["eta_x"], self.problem._optimiser_params["eta_y"]
        Fx, Fy = self.problem.grad(self.x, self.y)
        return (eta_x*LA.norm(Fx)**2 + eta_y*LA.norm(Fy)**2) <= self.eps * self.grad_norm_0 ** 2

    def stopping_criteria_grad_absolute(self):
        eta_x, eta_y = self.problem._optimiser_params["eta_x"], self.problem._optimiser_params["eta_y"]
        Fx, Fy = self.problem.grad(self.x, self.y)
        return (eta_x*LA.norm(Fx)**2 + eta_y*LA.norm(Fy)**2) <= self.eps
    
    def stopping_criteria_loss(self):
        return self.problem.loss(self.x, self.y) <= self.eps
    
    def stopping_criteria_sliding(self):
        # TODO: calculate grad truely. Now it's only for quadratic case
        # return np.sum(self.problem.grad(self.x, self.y)[0]**2) <= 0.5*self.problem.L_x**2*np.sum((self.x_hist[0] - x_opt)**2)
        _A = self.problem.A
        _C = self.problem.C
        f_grad_x, _ = self.problem.fg_grads(self.x, self.y)
        x_opt = LA.solve(3/4*_A.T.dot(LA.inv(_C)).dot(_A) + 
                         1/self.problem._optimiser_params["theta"]*np.eye(self.x.shape[0]),
                         self.problem._optimiser_params["right_part"])
        return np.sum(LA.norm(f_grad_x + 3/4 * _A.T.dot(LA.inv(_C)).dot(_A).dot(self.x))**2) <= 0.5*self.problem.L_x**2*np.sum((self.x_hist[0] - x_opt)**2)
    
    def stopping_criteria_acceg_sliding(self):
        eta_x, eta_y = self.problem._optimiser_params["eta_x"], self.problem._optimiser_params["eta_y"]
        f_grad_x, g_grad_y = self.problem.fg_grads(self.x, self.y)
        Fx, Fy = self.problem.grad(self.x, self.y)
        return (eta_x*LA.norm(Fx)**2 + eta_y*LA.norm(Fy)**2) <= 1/3*(1/eta_x* LA.norm(self.x - self.x_hist[0])**2 + 
                                                                     1/eta_y* LA.norm(self.y - self.y_hist[0])**2)

    def stopping_criteria_none(self):
        return False
