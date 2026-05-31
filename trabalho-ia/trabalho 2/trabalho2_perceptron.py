# Trabalho 2 - Inteligência Artificial
# Uso do aprendizado (algoritmo Perceptron) para classificar
# notas de alunos em duas classes linearmente separáveis.
#
# Exemplo: Classificação de Aprovação/Reprovação por Notas
#   x1 = nota da prova    (normalizada: nota_prova  / 10)
#   x2 = nota do trabalho (normalizada: nota_trab   / 10)
#   Classe 0 = Reprovado  (notas baixas → x1+x2 ≈ 0.35)
#   Classe 1 = Aprovado   (notas altas  → x1+x2 ≈ 1.60)
#
# Normalização: divide por 10 (escala de 0 a 10 → 0 a 1)
# Referência: adaptado do código de aula do prof. Mauricio C. Mario

import numpy as np
import matplotlib.pyplot as plt


# ─── FUNÇÕES DO PERCEPTRON ────────────────────────────────────────────────────

def degrau(x):
    """Função de ativação de limiar (θ = 0): retorna 1 se x >= 0, senão 0."""
    return 1 if x >= 0 else 0


def saida_perceptron(pesos, entradas):
    y = np.dot(pesos, entradas)
    return degrau(y)


def ajustes(sinapses, entradas, saida):
    taxa_aprendizagem = 0.08
    saida_parcial = saida_perceptron(sinapses, entradas)
    for j in range(3):
        sinapses[j] = sinapses[j] + taxa_aprendizagem * (saida[0] - saida_parcial) * entradas[j]
    saida = saida_parcial
    return sinapses, saida


def teste_generalizacao(sinapses, entradas, saida):
    saida_parcial = saida_perceptron(sinapses, entradas)
    saida = saida_parcial
    return sinapses, saida


# ─── DADOS DE TREINAMENTO ────────────────────────────────────────────────────
# Formato: [-1 (bias), x1 (prova/10), x2 (trabalho/10)]
#
# Tabela de dados brutos (antes da normalização):
#  Elemento | Prova | Trabalho | Classe
#  ---------|-------|----------|-------
#     1     |  2.0  |   1.5    |   0
#     2     |  1.5  |   2.0    |   0
#     3     |  2.5  |   1.0    |   0
#     4     |  8.0  |   7.0    |   1
#     5     |  7.0  |   9.0    |   1
#     6     |  9.0  |   8.0    |   1

neuronio = [0.22, -0.33, 0.44]   # pesos iniciais das sinapses do neurônio

# Classe 0 = Reprovado
padrao_0_0 = [-1, 0.20, 0.15]    # prova=2.0, trabalho=1.5
padrao_0_1 = [-1, 0.15, 0.20]    # prova=1.5, trabalho=2.0
padrao_0_2 = [-1, 0.25, 0.10]    # prova=2.5, trabalho=1.0

# Classe 1 = Aprovado
padrao_1_0 = [-1, 0.80, 0.70]    # prova=8.0, trabalho=7.0
padrao_1_1 = [-1, 0.70, 0.90]    # prova=7.0, trabalho=9.0
padrao_1_2 = [-1, 0.90, 0.80]    # prova=9.0, trabalho=8.0

saida0 = [0]   # rótulo Classe 0 (reprovado)
saida1 = [1]   # rótulo Classe 1 (aprovado)


# ─── TREINAMENTO: 11 ciclos ───────────────────────────────────────────────────
n = 0
for _ in range(11):
    neuronio, saida_0 = ajustes(neuronio, padrao_0_0, saida0)
    print(neuronio, "saida0 =", saida_0)
    neuronio, saida_0 = ajustes(neuronio, padrao_0_1, saida0)
    print(neuronio, "saida0 =", saida_0)
    neuronio, saida_0 = ajustes(neuronio, padrao_0_2, saida0)
    print(neuronio, "saida0 =", saida_0)
    neuronio, saida_1 = ajustes(neuronio, padrao_1_0, saida1)
    print(neuronio, "saida1 =", saida_1)
    neuronio, saida_1 = ajustes(neuronio, padrao_1_1, saida1)
    print(neuronio, "saida1 =", saida_1)
    neuronio, saida_1 = ajustes(neuronio, padrao_1_2, saida1)
    print(neuronio, "saida1 =", saida_1)
    n = n + 1
    print("número de ciclos =", n)

print("\nPesos finais após treinamento:", neuronio)


# ─── GRÁFICO 1: SEPARAÇÃO DE CLASSES (TREINAMENTO) ───────────────────────────
# Reta separadora calculada a partir dos pesos convergidos:
#   dot(neuronio, entradas) = 0
#   neuronio[0]*(-1) + neuronio[1]*x1 + neuronio[2]*x2 = 0
#   x2 = (neuronio[0] - neuronio[1]*x1) / neuronio[2]

x_reta = np.linspace(0.0, 1.0, 50)
y_reta = (neuronio[0] - neuronio[1] * x_reta) / neuronio[2]

# Coordenadas dos pontos de treinamento
x_c0 = [0.20, 0.15, 0.25]
y_c0 = [0.15, 0.20, 0.10]
x_c1 = [0.80, 0.70, 0.90]
y_c1 = [0.70, 0.90, 0.80]

plt.figure(figsize=(8, 6))
plt.scatter(x_c0, y_c0, color='blue',   marker='s', s=120, label='Reprovado – Classe 0', zorder=5)
plt.scatter(x_c1, y_c1, color='orange', marker='o', s=120, label='Aprovado  – Classe 1', zorder=5)
plt.plot(x_reta, y_reta, color='green', marker='*', linestyle='--', label='Separação linear', zorder=3)
plt.title("Separação de Classes com Perceptron\nClassificação de Notas: Aprovado vs Reprovado")
plt.xlabel("Nota da Prova (normalizada  x1 = nota/10)")
plt.ylabel("Nota do Trabalho (normalizada  x2 = nota/10)")
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.legend()
plt.grid(True)
plt.savefig("grafico_item3_treinamento.png", dpi=150, bbox_inches='tight')
plt.show()


# ─── TESTE DE GENERALIZAÇÃO ───────────────────────────────────────────────────
# Pontos novos que NÃO participaram do treinamento
#
#  Teste | Prova | Trabalho | Esperado
#  ------|-------|----------|----------
#    0   |  1.0  |   2.0    | Classe 0
#    1   |  7.5  |   8.5    | Classe 1
#    2   |  3.0  |   1.5    | Classe 0
#    3   |  8.5  |   7.5    | Classe 1
#    4   |  2.0  |   2.5    | Classe 0
#    5   |  6.5  |   8.0    | Classe 1

padrao_teste_0 = [-1, 0.10, 0.20]   # prova=1.0, trabalho=2.0  → esperado: Classe 0
padrao_teste_1 = [-1, 0.75, 0.85]   # prova=7.5, trabalho=8.5  → esperado: Classe 1
padrao_teste_2 = [-1, 0.30, 0.15]   # prova=3.0, trabalho=1.5  → esperado: Classe 0
padrao_teste_3 = [-1, 0.85, 0.75]   # prova=8.5, trabalho=7.5  → esperado: Classe 1
padrao_teste_4 = [-1, 0.20, 0.25]   # prova=2.0, trabalho=2.5  → esperado: Classe 0
padrao_teste_5 = [-1, 0.65, 0.80]   # prova=6.5, trabalho=8.0  → esperado: Classe 1

print("\nteste de generalização")
neuronio, saida_0 = teste_generalizacao(neuronio, padrao_teste_0, saida0)
print(neuronio, "saida0 =", saida_0)
neuronio, saida_1 = teste_generalizacao(neuronio, padrao_teste_1, saida1)
print(neuronio, "saida1 =", saida_1)
neuronio, saida_0 = teste_generalizacao(neuronio, padrao_teste_2, saida0)
print(neuronio, "saida0 =", saida_0)
neuronio, saida_1 = teste_generalizacao(neuronio, padrao_teste_3, saida1)
print(neuronio, "saida1 =", saida_1)
neuronio, saida_0 = teste_generalizacao(neuronio, padrao_teste_4, saida0)
print(neuronio, "saida0 =", saida_0)
neuronio, saida_1 = teste_generalizacao(neuronio, padrao_teste_5, saida1)
print(neuronio, "saida1 =", saida_1)


# ─── GRÁFICO 2: TREINAMENTO + GENERALIZAÇÃO ──────────────────────────────────
x_tst_c0 = [0.10, 0.30, 0.20]
y_tst_c0 = [0.20, 0.15, 0.25]
x_tst_c1 = [0.75, 0.85, 0.65]
y_tst_c1 = [0.85, 0.75, 0.80]

plt.figure(figsize=(8, 6))
plt.scatter(x_c0,     y_c0,     color='blue',   marker='s', s=120, label='Treinamento – Classe 0', zorder=5)
plt.scatter(x_c1,     y_c1,     color='orange', marker='o', s=120, label='Treinamento – Classe 1', zorder=5)
plt.scatter(x_tst_c0, y_tst_c0, color='cyan',   marker='^', s=120, label='Generalização – Classe 0', zorder=5)
plt.scatter(x_tst_c1, y_tst_c1, color='red',    marker='^', s=120, label='Generalização – Classe 1', zorder=5)
plt.plot(x_reta, y_reta, color='green', marker='*', linestyle='--', label='Separação linear', zorder=3)
plt.title("Generalização com Perceptron\nClassificação de Notas: Aprovado vs Reprovado")
plt.xlabel("Nota da Prova (normalizada  x1 = nota/10)")
plt.ylabel("Nota do Trabalho (normalizada  x2 = nota/10)")
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.legend()
plt.grid(True)
plt.show()
