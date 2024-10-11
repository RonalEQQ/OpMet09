import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# Definición de los coeficientes de la función objetivo
# Queremos maximizar Z = x1 + x2
# Pero linprog realiza minimización, así que minimizaremos -Z = -x1 -x2
c = [-1, -1]

# Definición de las restricciones
# 3x1 + 4x2 <= 24 (GPUs)
# 5x1 + 8x2 <= 50 (Tiempo)
A = [
    [3, 4],
    [5, 8]
]
b = [24, 50]

# Definición de los límites de las variables (x1, x2 >= 0)
x0_bounds = (0, None)
x1_bounds = (0, None)

# Resolución del problema de programación lineal
res = linprog(c, A_ub=A, b_ub=b, bounds=[x0_bounds, x1_bounds], method='highs')

# Extracción de los resultados
x1_opt, x2_opt = res.x
Z_opt = x1_opt + x2_opt

print("Solución Óptima:")
print(f"Entrenamientos del Modelo 1 (x1): {x1_opt:.2f}")
print(f"Entrenamientos del Modelo 2 (x2): {x2_opt:.2f}")
print(f"Número Total de Entrenamientos (Z): {Z_opt:.2f}")

# Graficación de las restricciones y la región factible

# Definir los valores para x1
x1 = np.linspace(0, 10, 400)

# Restricción de GPUs: 3x1 + 4x2 <= 24 => x2 = (24 - 3x1)/4
x2_gpu = (24 - 3 * x1) / 4

# Restricción de Tiempo: 5x1 + 8x2 <= 50 => x2 = (50 - 5x1)/8
x2_tiempo = (50 - 5 * x1) / 8

# Graficar las líneas de las restricciones
plt.figure(figsize=(10, 8))
plt.plot(x1, x2_gpu, label=r'$3x_1 + 4x_2 \leq 24$ (GPUs)', color='blue')
plt.plot(x1, x2_tiempo, label=r'$5x_1 + 8x_2 \leq 50$ (Tiempo)', color='green')

# Rellenar la región factible
# Encontrar puntos de intersección con los ejes
# Restricción de GPUs
x1_gpu = 24 / 3
x2_gpu_axis = 24 / 4

# Restricción de Tiempo
x1_tiempo = 50 / 5
x2_tiempo_axis = 50 / 8

# Puntos de intersección factibles
vertices = [
    (0, 0),
    (x1_gpu, 0),
    (0, x2_gpu_axis)
]

# Definir la región factible
x_fill = [0, x1_gpu, 0]
y_fill = [0, 0, x2_gpu_axis]
plt.fill_between(x1, np.minimum(x2_gpu, x2_tiempo), where=(x2_gpu >=0) & (x2_tiempo >=0), color='grey', alpha=0.3, label='Región Factible')

# Graficar el punto óptimo
plt.plot(x1_opt, x2_opt, 'ro', label='Solución Óptima')

# Anotar el punto óptimo
plt.annotate(f'Óptimo ({x1_opt:.2f}, {x2_opt:.2f})', 
             xy=(x1_opt, x2_opt), 
             xytext=(x1_opt + 0.5, x2_opt + 0.5),
             arrowprops=dict(facecolor='black', shrink=0.05))

# Configuración del gráfico
plt.xlim(0, max(x1_gpu, x1_tiempo) + 1)
plt.ylim(0, max(x2_gpu_axis, x2_tiempo_axis) + 1)
plt.xlabel('$x_1$ (Entrenamientos Modelo 1)')
plt.ylabel('$x_2$ (Entrenamientos Modelo 2)')
plt.title('Región Factible y Solución Óptima')
plt.legend()
plt.grid(True)
plt.show()
