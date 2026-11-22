"""
metodos.py
==========
Implementación de dos métodos numéricos de un paso para resolver problemas
de valor inicial (PVI) de la forma

        y'(t) = f(t, y),    y(t0) = y0,

donde y puede ser un escalar (ED de primer orden) o un vector (ED de segundo
orden reducida a sistema, o sistema de EDs). Un mismo código sirve para todos
los casos porque se trabaja siempre con arreglos de NumPy.

Métodos:
    - heun : Heun (Euler mejorado), segundo orden.
    - rk3  : Runge-Kutta clásico de Kutta, tercer orden.

Para reducir una ED de segundo orden  x'' = g(t, x, x')  a sistema se define
v = x', de modo que  y = (x, v)  y  f(t, y) = (v, g(t, x, v)).
"""

import numpy as np


def _preparar(t0, tf, y0, h):
    """
    Valida los datos y reserva la memoria para la solución.

    Parámetros
    ----------
    t0, tf : float
        Extremos del intervalo de integración (tf > t0).
    y0 : float o array_like
        Condición inicial.
    h : float
        Tamaño de paso (h > 0). Debe dividir exactamente a (tf - t0).

    Retorna
    -------
    n : int
        Número de pasos.
    t : ndarray, forma (n+1,)
        Nodos de tiempo  t_k = t0 + k h.
    y : ndarray, forma (n+1, m)
        Arreglo donde se guarda la solución; la fila 0 es y0.
    """
    if h <= 0:
        raise ValueError("El tamaño de paso h debe ser positivo.")
    if tf <= t0:
        raise ValueError("Se requiere tf > t0.")

    n = int(round((tf - t0) / h))
    if abs(n * h - (tf - t0)) > 1e-9 * max(1.0, abs(tf - t0)):
        raise ValueError("h debe dividir exactamente al intervalo [t0, tf].")

    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    # Los nodos se calculan como t0 + k*h (y no sumando h repetidamente)
    # para no acumular error de redondeo en el tiempo.
    t = t0 + h * np.arange(n + 1)
    y = np.empty((n + 1, y0.size))
    y[0] = y0
    return n, t, y


def heun(f, t0, tf, y0, h):
    """
    Método de Heun (Euler mejorado), de segundo orden.

    En cada paso se hace una predicción con Euler y luego se corrige con el
    promedio de las pendientes en los dos extremos del paso:

        k1      = f(t_n, y_n)
        k2      = f(t_n + h, y_n + h*k1)           (pendiente con la predicción)
        y_{n+1} = y_n + (h/2) * (k1 + k2)

    Usa 2 evaluaciones de f por paso. Error local O(h^3), error global O(h^2).

    Parámetros
    ----------
    f : callable
        Función f(t, y) que devuelve y' como arreglo de NumPy (misma forma
        que y).
    t0, tf : float
        Extremos del intervalo.
    y0 : float o array_like
        Condición inicial y(t0).
    h : float
        Tamaño de paso.

    Retorna
    -------
    t : ndarray, forma (n+1,)
        Nodos de tiempo.
    y : ndarray, forma (n+1, m)
        Aproximación de la solución; y[k] aproxima y(t[k]).
    """
    n, t, y = _preparar(t0, tf, y0, h)
    for k in range(n):
        k1 = np.asarray(f(t[k], y[k]), dtype=float)
        k2 = np.asarray(f(t[k] + h, y[k] + h * k1), dtype=float)
        y[k + 1] = y[k] + 0.5 * h * (k1 + k2)
    return t, y


def rk3(f, t0, tf, y0, h):
    """
    Método de Runge-Kutta de tercer orden (versión clásica de Kutta).

        k1      = f(t_n, y_n)
        k2      = f(t_n + h/2, y_n + (h/2)*k1)
        k3      = f(t_n + h,   y_n - h*k1 + 2*h*k2)
        y_{n+1} = y_n + (h/6) * (k1 + 4*k2 + k3)

    Usa 3 evaluaciones de f por paso. Error local O(h^4), error global O(h^3).

    Parámetros
    ----------
    f : callable
        Función f(t, y) que devuelve y' como arreglo de NumPy.
    t0, tf : float
        Extremos del intervalo.
    y0 : float o array_like
        Condición inicial y(t0).
    h : float
        Tamaño de paso.

    Retorna
    -------
    t : ndarray, forma (n+1,)
        Nodos de tiempo.
    y : ndarray, forma (n+1, m)
        Aproximación de la solución; y[k] aproxima y(t[k]).
    """
    n, t, y = _preparar(t0, tf, y0, h)
    for k in range(n):
        k1 = np.asarray(f(t[k], y[k]), dtype=float)
        k2 = np.asarray(f(t[k] + 0.5 * h, y[k] + 0.5 * h * k1), dtype=float)
        k3 = np.asarray(f(t[k] + h, y[k] - h * k1 + 2.0 * h * k2), dtype=float)
        y[k + 1] = y[k] + (h / 6.0) * (k1 + 4.0 * k2 + k3)
    return t, y
