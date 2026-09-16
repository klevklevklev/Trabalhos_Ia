import matplotlib
matplotlib.use('Agg')          # salva em arquivo sem abrir janela
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import os

# ─── Funcoes da rede neural ──────────────────────────────────────────────────

def sigmoid(t):
    return (2 / (1 + math.exp(-t))) - 1

def dot(v, w):
    return sum(vi * wi for vi, wi in zip(v, w))

def neuronio_MCP(pesos, entradas):
    return sigmoid(dot(pesos, entradas))

def feed_forward(rede_neural, vetor_entrada):
    vetor_saida = []
    for ponteiro in rede_neural:
        entrada_com_bias = vetor_entrada + [1]
        saida = [neuronio_MCP(neuronio, entrada_com_bias) for neuronio in ponteiro]
        vetor_saida.append(saida)
        vetor_entrada = saida
    return vetor_saida

def backpropagation(rede_neural, vetor_entrada, vetor_saida_desejada, alpha):
    saidas_intermediarias, saidas_neuronios = feed_forward(rede_neural, vetor_entrada)

    deltas_saida = [
        (0.5 * (1 + s) * (1 - s)) * (s - vetor_saida_desejada[i]) * alpha
        for i, s in enumerate(saidas_neuronios)
    ]

    for i, neuronio_saida in enumerate(rede_neural[-1]):
        for j, si in enumerate(saidas_intermediarias + [1]):
            neuronio_saida[j] -= deltas_saida[i] * si

    deltas_intermediarios = [
        0.5 * alpha * (1 + si) * (1 - si) *
        dot(deltas_saida, [n[i] for n in rede_neural[-1]])
        for i, si in enumerate(saidas_intermediarias)
    ]

    for i, neuronio_intermediario in enumerate(rede_neural[0]):
        for j, inp in enumerate(vetor_entrada + [1]):
            neuronio_intermediario[j] -= deltas_intermediarios[i] * inp

def funcao(x):
    val = math.sin(math.pi / 180 * x) * (-math.cos(math.pi / 180 * x))
    return [[val]]

def predict(entrada_teste, rede_neural):
    return feed_forward(rede_neural, entrada_teste)[-1]

# ─── Parametros ──────────────────────────────────────────────────────────────
CICLOS = 10           # ciclos fixos de treinamento
                      # 10 ciclos evidenciam as diferencas entre alpha e n;
                      # a funcao f(x)=-sin(2x)/2 converge muito rapidamente.
alphas       = [0.08, 0.1, 0.4]
neuronios_list = [3, 6, 8]

os.makedirs('graficos', exist_ok=True)

t = np.arange(0, 360, 1)
entrada_plot = []
for x in range(360):
    entrada_plot += funcao(x)[0]

# ─── Coleta de resultados ────────────────────────────────────────────────────
resultados = []   # (caso, alpha, n, mse, saida_rede)

caso = 1
for n_neuronios in neuronios_list:
    for alpha in alphas:

        random.seed(0)   # mesmo seed para o mesmo n -> isola o efeito de alpha
        camada_oculta = [
            [random.uniform(-0.5, 0.5) for _ in range(2)]
            for _ in range(n_neuronios)
        ]
        camada_saida = [
            [random.uniform(-0.5, 0.5) for _ in range(n_neuronios + 1)]
            for _ in range(1)
        ]
        rede_neural = [camada_oculta, camada_saida]

        for _ in range(CICLOS):
            for x in range(360):
                e = funcao(x)
                for ve, vs in zip(e, e):
                    backpropagation(rede_neural, ve, vs, alpha)

        saida_rede = []
        for x in range(360):
            e = funcao(x)
            for ve, vt in zip(e, e):
                saida_rede.extend(predict(vt, rede_neural))

        mse = sum((r - f) ** 2 for r, f in zip(saida_rede, entrada_plot)) / 360
        resultados.append((caso, alpha, n_neuronios, mse, saida_rede))
        print(f"Caso {caso} | alpha={alpha}  n={n_neuronios}  MSE={mse:.5f}")
        caso += 1

# ─── Graficos individuais ────────────────────────────────────────────────────
titulos = [
    f'Caso {c}: alpha={a}, n={n}  (MSE={mse:.4f})'
    for c, a, n, mse, _ in resultados
]

for (c, a, n, mse, saida_rede) in resultados:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, entrada_plot, color='blue',   linewidth=2,   label='f(x) original')
    ax.plot(t, saida_rede,   color='orange', linewidth=2,
            linestyle='--', label='Saida da rede')
    ax.set_title(f'Caso {c}:  alpha={a},  n={n}  |  MSE={mse:.4f}', fontsize=12)
    ax.set_xlabel('angulo (graus)')
    ax.set_ylabel('f(x) = sen(x)*(-cos(x))')
    ax.set_ylim(-0.8, 0.8)
    ax.grid(True, alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'graficos/caso_{c:02d}_alpha{str(a).replace(".","")}_n{n}.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> graficos/caso_{c:02d} salvo")

# ─── Painel geral 3x3 ────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
fig.suptitle(
    f'Aproximacao Funcional: f(x) = sen(x)*(-cos(x))   |   {CICLOS} ciclos',
    fontsize=13, fontweight='bold'
)

for idx, (c, a, n, mse, saida_rede) in enumerate(resultados):
    ax = axes[idx // 3][idx % 3]
    ax.plot(t, entrada_plot, color='blue',   linewidth=1.6, label='f(x)')
    ax.plot(t, saida_rede,   color='orange', linewidth=1.6,
            linestyle='--', label='rede')
    ax.set_title(f'Caso {c}: a={a}, n={n}  MSE={mse:.4f}', fontsize=9)
    ax.set_xlabel('angulo', fontsize=7)
    ax.set_ylabel('f(x)', fontsize=7)
    ax.set_ylim(-0.8, 0.8)
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig('graficos/painel_geral.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\nPainel geral salvo em graficos/painel_geral.png")
print(f"Ciclos de treinamento: {CICLOS}")
