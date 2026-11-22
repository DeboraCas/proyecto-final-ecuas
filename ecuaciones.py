import numpy as np


# ED de primer orden: calentamiento de un motor
# 100 T' + 2 (T - 25) = 20,   T(0) = 25,   0 <= t <= 200 s    Solución: T(t) = 35 - 10 e^{-t/50}
def _f_motor(t, y):
    T = y[0]
    return np.array([(20.0 - 2.0 * (T - 25.0)) / 100.0])


def _exacta_motor(t):
    t = np.asarray(t, dtype=float)
    return (35.0 - 10.0 * np.exp(-t / 50.0)).reshape(-1, 1)


MOTOR = dict(
    nombre="motor",
    f=_f_motor,
    t0=0.0,
    tf=200.0,
    y0=[25.0],
    exacta=_exacta_motor,
    etiquetas=["T(t) [°C]"],
)


# ED de segundo orden: elevador
#    m x'' + c x' = F_m - m g,   x(0) = 0,  x'(0) = 0,   0 <= t <= 10 s    Solución: v(t) = 5 (1 - e^{-t/5}),   x(t) = 5 t + 25 e^{-t/5} - 25
M_ELEV, C_ELEV, FM_ELEV, G = 500.0, 100.0, 5405.0, 9.81


def _f_elevador(t, y):
    x, v = y
    return np.array([v, (FM_ELEV - M_ELEV * G - C_ELEV * v) / M_ELEV])


def _exacta_elevador(t):
    t = np.asarray(t, dtype=float)
    x = 5.0 * t + 25.0 * np.exp(-t / 5.0) - 25.0
    v = 5.0 * (1.0 - np.exp(-t / 5.0))
    return np.column_stack([x, v])


ELEVADOR = dict(
    nombre="elevador",
    f=_f_elevador,
    t0=0.0,
    tf=10.0,
    y0=[0.0, 0.0],
    exacta=_exacta_elevador,
    etiquetas=["x(t) [m]", "x'(t) [m/s]"],
)


# Sistema lineal 2x2: motor DC
#    i' = 24 - 2 i - 0.02 w,    w' = i - 10 w,    i(0) = w(0) = 0,
#    0 <= t <= 10 s.
#    Solución (por eliminación):  w'' + 12 w' + 20.02 w = 24
#      m1,2 = -6 +- s,  s = sqrt(15.98),   w_p = 24/20.02
#      w = w_p + C1 e^{m1 t} + C2 e^{m2 t},   i = w' + 10 w
def _f_motor_dc(t, y):
    i, w = y
    return np.array([24.0 - 2.0 * i - 0.02 * w, i - 10.0 * w])


def _exacta_motor_dc(t):
    t = np.asarray(t, dtype=float)
    s = np.sqrt(15.98)
    m1, m2 = -6.0 + s, -6.0 - s
    wp = 24.0 / 20.02
    C1 = -(wp / 2.0) * (1.0 + 6.0 / s)
    C2 = -(wp / 2.0) * (1.0 - 6.0 / s)
    w = wp + C1 * np.exp(m1 * t) + C2 * np.exp(m2 * t)
    i = 10.0 * wp + C1 * (10.0 + m1) * np.exp(m1 * t) \
        + C2 * (10.0 + m2) * np.exp(m2 * t)
    return np.column_stack([i, w])


MOTOR_DC = dict(
    nombre="motor_dc",
    f=_f_motor_dc,
    t0=0.0,
    tf=10.0,
    y0=[0.0, 0.0],
    exacta=_exacta_motor_dc,
    etiquetas=["i(t) [A]", "ω(t) [rad/s]"],
)


# Sistema NO lineal 2x2: Lotka-Volterra (sin solución analítica)
#    x' = a x - b x y,    y' = d x y - c y,    x(0) = 10,  y(0) = 5
#    Cantidad conservada:  H(x, y) = d x - c ln x + b y - a ln y

A_LV, B_LV, C_LV, D_LV = 1.1, 0.4, 0.4, 0.1


def _f_lv(t, y):
    x, p = y
    return np.array([A_LV * x - B_LV * x * p, D_LV * x * p - C_LV * p])


def H_lv(y):
    """Cantidad conservada de Lotka-Volterra. y tiene forma (n, 2)."""
    x, p = y[:, 0], y[:, 1]
    return D_LV * x - C_LV * np.log(x) + B_LV * p - A_LV * np.log(p)


LOTKA_VOLTERRA = dict(
    nombre="lotka_volterra",
    f=_f_lv,
    t0=0.0,
    tf=20.0,
    y0=[10.0, 5.0],
    exacta=None,
    etiquetas=["x(t) presas", "y(t) depredadores"],
)
