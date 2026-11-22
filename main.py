"""
Pruebas de Heun y RK3 para el proyecto final.

Resuelve los cuatro problemas con los dos métodos, calcula los errores y
guarda las figuras en figs/ y las tablas en tablas/.

"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np

import ecuaciones as eq
from metodos import heun, rk3

METODOS = {"Heun": heun, "RK3": rk3}
COLORES = {"Heun": "tab:orange", "RK3": "tab:green"}

# así figs/ y tablas/ quedan junto a este archivo, se corra desde donde se corra
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("figs", exist_ok=True)
os.makedirs("tablas", exist_ok=True)


def fmt_tex(x):
    if x == 0:
        return "0"
    mantisa, exp = f"{x:.2e}".split("e")
    return rf"${mantisa}\times10^{{{int(exp)}}}$"


def orden_observado(hs, errores):
    return np.polyfit(np.log(hs), np.log(errores), 1)[0]


def resolver(problema, metodo, h):
    return METODOS[metodo](problema["f"], problema["t0"], problema["tf"],
                           problema["y0"], h)


def error_max(problema, metodo, h):
    """Mayor diferencia absoluta entre la solución numérica y la exacta."""
    t, y = resolver(problema, metodo, h)
    return np.max(np.abs(y - problema["exacta"](t)))


def escribir_tabla(nombre, titulo, etiqueta, hs, errs, ordenes):
    filas = "\n".join(
        f"{h:g} & {fmt_tex(errs['Heun'][k])} & {fmt_tex(errs['RK3'][k])} \\\\"
        for k, h in enumerate(hs)
    )
    texto = (
        "\\begin{table}[H]\n\\centering\n"
        f"\\caption{{{titulo}}}\n\\label{{{etiqueta}}}\n"
        "\\begin{tabular}{ccc}\n\\toprule\n"
        "$h$ & Error máx. Heun & Error máx. RK3 \\\\\n\\midrule\n"
        f"{filas}\n\\midrule\n"
        f"Orden observado & {ordenes['Heun']:.2f} & {ordenes['RK3']:.2f} \\\\\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    with open(f"tablas/{nombre}.tex", "w", encoding="utf-8") as fh:
        fh.write(texto)


def verificar_analiticas():
    """Revisa que las soluciones exactas cumplan la condición inicial y la ED."""
    print("=" * 70)
    print("VERIFICACIÓN DE LAS SOLUCIONES ANALÍTICAS")
    print("=" * 70)
    delta = 1e-5
    for p in (eq.MOTOR, eq.ELEVADOR, eq.MOTOR_DC):
        t = np.linspace(p["t0"] + 1e-3, p["tf"] - 1e-3, 400)
        y = p["exacta"](t)
        # derivada de la exacta (diferencia central) contra f(t, y)
        dy = (p["exacta"](t + delta) - p["exacta"](t - delta)) / (2 * delta)
        fy = np.array([p["f"](tk, yk) for tk, yk in zip(t, y)])
        residuo = np.max(np.abs(dy - fy))
        ci = np.max(np.abs(p["exacta"](np.array([p["t0"]]))[0] - p["y0"]))
        print(f"{p['nombre']:<10}  |y(t0)-y0| = {ci:.2e}   "
              f"residuo máx. de la ED = {residuo:.2e}")
    print()


# (problema, valores de h, h para la figura, título de la tabla)
PRUEBAS = [
    (eq.MOTOR, [10, 5, 2.5, 1, 0.5, 0.1], 10,
     "Calentamiento del motor: error máximo respecto de la solución analítica."),
    (eq.ELEVADOR, [1, 0.5, 0.25, 0.1, 0.05, 0.01], 1,
     "Elevador: error máximo respecto de la solución analítica."),
    (eq.MOTOR_DC, [0.1, 0.05, 0.025, 0.0125, 0.00625], 0.1,
     "Motor DC: error máximo respecto de la solución analítica."),
]


def probar_con_solucion_analitica(problema, hs, h_grueso, titulo):
    nombre = problema["nombre"]
    print("=" * 70)
    print(f"PROBLEMA: {nombre}   intervalo [{problema['t0']}, {problema['tf']}]")
    print("=" * 70)

    errs = {m: [error_max(problema, m, h) for h in hs] for m in METODOS}
    ordenes = {m: orden_observado(hs, errs[m]) for m in METODOS}

    print(f"{'h':>10} {'Heun':>14} {'RK3':>14}")
    for k, h in enumerate(hs):
        print(f"{h:>10g} {errs['Heun'][k]:>14.3e} {errs['RK3'][k]:>14.3e}")
    print(f"Orden observado: Heun = {ordenes['Heun']:.2f}, "
          f"RK3 = {ordenes['RK3']:.2f}\n")

    escribir_tabla(f"tabla_{nombre}", titulo, f"tab:{nombre}",
                   hs, errs, ordenes)

    # figura con un h grande, para que se vean los puntos de cada método
    n_comp = len(problema["y0"])
    t_fino = np.linspace(problema["t0"], problema["tf"], 800)
    y_fino = problema["exacta"](t_fino)
    fig, ejes = plt.subplots(1, n_comp, figsize=(5.5 * n_comp, 4),
                             squeeze=False)
    for j in range(n_comp):
        ax = ejes[0, j]
        ax.plot(t_fino, y_fino[:, j], "k-", lw=1.8, label="Analítica")
        for m in METODOS:
            t, y = resolver(problema, m, h_grueso)
            ax.plot(t, y[:, j], "o--", ms=4, color=COLORES[m],
                    label=f"{m} (h={h_grueso:g})")
        ax.set_xlabel("t [s]")
        ax.set_ylabel(problema["etiquetas"][j])
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(f"figs/sol_{nombre}.pdf")
    plt.close(fig)


def probar_lotka_volterra():
    p = eq.LOTKA_VOLTERRA
    hs = [0.2, 0.1, 0.05, 0.025, 0.0125]
    h_ref = 0.0005
    h_fig = 0.1
    print("=" * 70)
    print(f"PROBLEMA NO LINEAL: Lotka-Volterra   intervalo "
          f"[{p['t0']}, {p['tf']}]")
    print("=" * 70)

    # referencia: RK3 con un h muy chico (sigue siendo numérica, no exacta)
    t_ref, y_ref = resolver(p, "RK3", h_ref)
    H0 = eq.H_lv(np.array([p["y0"]]))[0]

    dif_h, dif_ref, deriva_H = {}, {}, {}
    for m in METODOS:
        dif_h[m], dif_ref[m], deriva_H[m] = [], [], []
        for h in hs:
            t1, y1 = resolver(p, m, h)
            t2, y2 = resolver(p, m, h / 2)
            # h contra h/2, en los tiempos que tienen en común
            dif_h[m].append(np.max(np.abs(y1 - y2[::2])))
            # contra la referencia: cada nodo de h cae cada `paso` nodos de ella
            paso = int(round(h / h_ref))
            dif_ref[m].append(np.max(np.abs(y1 - y_ref[::paso])))
            # cuánto se aleja H de su valor inicial
            deriva_H[m].append(np.max(np.abs(eq.H_lv(y1) - H0)))

    o_h = {m: orden_observado(hs, dif_h[m]) for m in METODOS}
    o_ref = {m: orden_observado(hs, dif_ref[m]) for m in METODOS}
    o_H = {m: orden_observado(hs, deriva_H[m]) for m in METODOS}

    print(f"H(0) = {H0:.6f}")
    print(f"{'h':>9} | {'|y_h - y_h/2| Heun':>20} {'RK3':>11} | "
          f"{'vs. ref. Heun':>14} {'RK3':>11} | "
          f"{'max|H-H0| Heun':>15} {'RK3':>11}")
    for k, h in enumerate(hs):
        print(f"{h:>9g} | {dif_h['Heun'][k]:>20.3e} {dif_h['RK3'][k]:>11.3e} | "
              f"{dif_ref['Heun'][k]:>14.3e} {dif_ref['RK3'][k]:>11.3e} | "
              f"{deriva_H['Heun'][k]:>15.3e} {deriva_H['RK3'][k]:>11.3e}")
    print("Órdenes observados (Heun, RK3):")
    print(f"  h vs h/2        : {o_h['Heun']:.2f}, {o_h['RK3']:.2f}")
    print(f"  vs. referencia  : {o_ref['Heun']:.2f}, {o_ref['RK3']:.2f}")
    print(f"  deriva de H     : {o_H['Heun']:.2f}, {o_H['RK3']:.2f}\n")

    filas = "\n".join(
        f"{h:g} & {fmt_tex(dif_ref['Heun'][k])} & {fmt_tex(dif_ref['RK3'][k])}"
        f" & {fmt_tex(deriva_H['Heun'][k])} & {fmt_tex(deriva_H['RK3'][k])}"
        " \\\\"
        for k, h in enumerate(hs)
    )
    texto = (
        "\\begin{table}[H]\n\\centering\n"
        "\\caption{Lotka-Volterra: diferencia máxima respecto de la solución "
        "de referencia ($h=0.0005$, RK3) y desviación máxima de $H$ respecto "
        "de $H(0)$.}\n\\label{tab:lotka_volterra}\n"
        "\\begin{tabular}{ccccc}\n\\toprule\n"
        " & \\multicolumn{2}{c}{Diferencia con la referencia} & "
        "\\multicolumn{2}{c}{$\\max|H-H(0)|$} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
        "$h$ & Heun & RK3 & Heun & RK3 \\\\\n\\midrule\n"
        f"{filas}\n\\midrule\n"
        f"Orden observado & {o_ref['Heun']:.2f} & {o_ref['RK3']:.2f} & "
        f"{o_H['Heun']:.2f} & {o_H['RK3']:.2f} \\\\\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    with open("tablas/tabla_lotka_volterra.tex", "w", encoding="utf-8") as fh:
        fh.write(texto)

    # plano de fase
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(y_ref[:, 0], y_ref[:, 1], "k-", lw=1.8, label="Referencia")
    for m in METODOS:
        t, y = resolver(p, m, h_fig)
        ax.plot(y[:, 0], y[:, 1], "--", color=COLORES[m],
                label=f"{m} (h={h_fig})")
    ax.plot(*p["y0"], "ko", label="Condición inicial")
    ax.set_xlabel("x (presas)")
    ax.set_ylabel("y (depredadores)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("figs/fase_lotka_volterra.pdf")
    plt.close(fig)

    # convergencia: contra la referencia (izquierda) y desviación de H (derecha)
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4))
    for m in METODOS:
        ejes[0].loglog(hs, dif_ref[m], "o-", color=COLORES[m],
                       label=f"{m} (pendiente ≈ {o_ref[m]:.2f})")
        ejes[1].loglog(hs, deriva_H[m], "o-", color=COLORES[m],
                       label=f"{m} (pendiente ≈ {o_H[m]:.2f})")
    ejes[0].set_ylabel("Diferencia máx. con la referencia")
    ejes[1].set_ylabel("max |H(t) - H(0)|")
    for ax in ejes:
        ax.set_xlabel("h")
        ax.grid(alpha=0.3, which="both")
        ax.set_xticks(hs)
        ax.set_xticklabels([f"{h:g}" for h in hs])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.legend()
    fig.tight_layout()
    fig.savefig("figs/convergencia_lotka_volterra.pdf")
    plt.close(fig)


if __name__ == "__main__":
    verificar_analiticas()
    for prob, hs, h_g, tit in PRUEBAS:
        probar_con_solucion_analitica(prob, hs, h_g, tit)
    probar_lotka_volterra()
    print("Listo. Figuras en figs/ y tablas en tablas/.")
