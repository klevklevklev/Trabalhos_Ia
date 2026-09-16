# -*- coding: utf-8 -*-
import math
import random

# ─── substituicao do modulo linear_algebra do livro ──────────────────────────
def dot(v, w):
    return sum(vi * wi for vi, wi in zip(v, w))

# ─── Funcoes da rede neural (sigmóide UNIPOLAR, igual ao codigo do professor) ─

def sigmoid(t):
    return 1 / (1 + math.exp(-t))

def neuronio_MCP(pesos, entradas):
    return sigmoid(dot(pesos, entradas))

def feed_forward(rede, vetor_entrada):
    vetor_saida = []
    for ponteiro in rede:
        entrada_com_bias = vetor_entrada + [1]
        saida = [neuronio_MCP(neuronio, entrada_com_bias) for neuronio in ponteiro]
        vetor_saida.append(saida)
        vetor_entrada = saida
    return vetor_saida

alpha = 0.99

def backpropagation(rede_neural, vetor_entrada, vetor_saida):
    saidas_intermediarias, saidas_neuronios = feed_forward(rede_neural, vetor_entrada)

    # delta camada de saida: derivada sigmoide unipolar * erro
    deltas_saida = [
        saida * (1 - saida) * (saida - vetor_saida[i]) * alpha
        for i, saida in enumerate(saidas_neuronios)
    ]

    # atualiza pesos camada de saida
    for i, neuronio_saida in enumerate(rede_neural[-1]):
        for j, saida_intermediaria in enumerate(saidas_intermediarias + [1]):
            neuronio_saida[j] -= deltas_saida[i] * saida_intermediaria

    # delta camada intermediaria: derivada sigmoide * retropropagacao
    deltas_intermediarios = [
        saida_intermediaria * (1 - saida_intermediaria) *
        dot(deltas_saida, [n[i] for n in rede_neural[-1]])
        for i, saida_intermediaria in enumerate(saidas_intermediarias)
    ]

    # atualiza pesos camada intermediaria
    for i, neuronio_intermediario in enumerate(rede_neural[0]):
        for j, entrada in enumerate(vetor_entrada + [1]):
            neuronio_intermediario[j] -= deltas_intermediarios[i] * entrada


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Padroes das vogais em grade 5x5 (1=pixel ativo, .=inativo) ───────────
    #    Adaptacao de "caracteres_numericos" para "caracteres_vogais"
    lista_treino = [
        """.111.
1...1
11111
1...1
1...1""",   # A

        """11111
1....
1111.
1....
11111""",   # E

        """11111
..1..
..1..
..1..
11111""",   # I

        """.111.
1...1
1...1
1...1
.111.""",   # O

        """1...1
1...1
1...1
1...1
.111.""",   # U
    ]

    def make_digit(dados_treino):
        return [1 if c == '1' else 0
                for row in dados_treino.split("\n")
                for c in row.strip()]

    # =========================================================
    # EVIDENCIA 1: adaptacao de caracteres_numericos, saidas
    #              e dimensao_saida para as vogais
    # =========================================================

    # 1a) caracteres_vogais
    caracteres_vogais = list(map(make_digit, lista_treino))
    print("\n" + "="*55)
    print("EVIDENCIA 1 - caracteres_vogais (vetores de entrada)")
    print("="*55)
    nomes = ['A', 'E', 'I', 'O', 'U']
    for nome, vetor in zip(nomes, caracteres_vogais):
        print(f"  Vogal {nome}: {vetor}")

    # 1b) saidas  
    saidas = [[1 if i == j else 0 for i in range(5)]
              for j in range(5)]
    print("\nEVIDENCIA 1 - saidas (matriz identidade 5x5):")
    for nome, s in zip(nomes, saidas):
        print(f"  Vogal {nome}: {s}")

    # 1c) dimensao_saida
    random.seed(0)
    dimensao_entrada  = 25
    neuronios_ocultos = 5
    dimensao_saida    = 5    # adaptado: era 10 (digitos), agora 5 (vogais)
    print(f"\nEVIDENCIA 1 - dimensao_saida = {dimensao_saida}  "
          f"(5 vogais: A E I O U)")

    camada_intermediaria = [
        [random.random() for _ in range(dimensao_entrada + 1)]
        for _ in range(neuronios_ocultos)
    ]
    camada_saida = [
        [random.random() for _ in range(neuronios_ocultos + 1)]
        for _ in range(dimensao_saida)
    ]
    rede_neural = [camada_intermediaria, camada_saida]

    # =========================================================
    # EVIDENCIA 2: treinamento das vogais de A ate U
    # =========================================================
    CICLOS = 10000
    print("\n" + "="*55)
    print(f"EVIDENCIA 2 - treinamento das vogais ({CICLOS} ciclos)")
    print("="*55)
    print(f"  Vogais treinadas : {nomes}")
    print(f"  Ciclos           : {CICLOS}")
    print(f"  Neuronios ocultos: {neuronios_ocultos}")
    print(f"  Dimensao entrada : {dimensao_entrada} (grade 5x5)")
    print(f"  Dimensao saida   : {dimensao_saida}")
    print("  Treinando...", end=" ", flush=True)

    for _ in range(CICLOS):
        for vetor_entrada, vetor_saida in zip(caracteres_vogais, saidas):
            backpropagation(rede_neural, vetor_entrada, vetor_saida)

    print("concluido.")

    def predict(vogal):
        return feed_forward(rede_neural, vogal)[-1]

    # =========================================================
    # EVIDENCIA 3: teste dos caracteres treinados
    # =========================================================
    print("\n" + "="*55)
    print("EVIDENCIA 3 - teste dos caracteres vogais treinados")
    print("="*55)
    print("  formato: [saida_A, saida_E, saida_I, saida_O, saida_U]")
    print()
    for i, vogal in enumerate(caracteres_vogais):
        saidas_rede = predict(vogal)
        reconhecido = nomes[saidas_rede.index(max(saidas_rede))]
        correto = "OK" if reconhecido == nomes[i] else "ERRO"
        print(f"  {nomes[i]}: {[round(p, 2) for p in saidas_rede]}"
              f"  -> reconhecido: {reconhecido}  [{correto}]")

    # =========================================================
    # EVIDENCIA 4: teste de generalizacao (variacoes com ruido)
    # =========================================================
    print("\n" + "="*55)
    print("EVIDENCIA 4 - generalizacao: variacoes das vogais")
    print("="*55)
    print("  Padroes distorcidos (1-2 pixels alterados)")
    print()

    variacoes = [
        ("A", ".1111\n1...1\n11111\n1...1\n1...1",
         "pixel extra na linha 1 (.1111 em vez de .111.)",
         [0,1,1,1,1, 1,0,0,0,1, 1,1,1,1,1, 1,0,0,0,1, 1,0,0,0,1]),

        ("E", "11111\n1....\n11111\n1....\n11111",
         "linha 3 completa (11111 em vez de 1111.)",
         [1,1,1,1,1, 1,0,0,0,0, 1,1,1,1,1, 1,0,0,0,0, 1,1,1,1,1]),

        ("I", ".1111\n..1..\n..1..\n..1..\n11111",
         "pixel a menos na linha 1 (.1111 em vez de 11111)",
         [0,1,1,1,1, 0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 1,1,1,1,1]),

        ("O", ".111.\n1..11\n1...1\n1...1\n.111.",
         "pixel extra na linha 2 (1..11 em vez de 1...1)",
         [0,1,1,1,0, 1,0,0,1,1, 1,0,0,0,1, 1,0,0,0,1, 0,1,1,1,0]),

        ("U", "1...1\n1...1\n1..11\n1...1\n.111.",
         "pixel extra na linha 3 (1..11 em vez de 1...1)",
         [1,0,0,0,1, 1,0,0,0,1, 1,0,0,1,1, 1,0,0,0,1, 0,1,1,1,0]),
    ]

    for vogal_ref, padrao, descricao, vetor in variacoes:
        print(f"  Variacao de {vogal_ref} ({descricao}):")
        for linha in padrao.split("\n"):
            print(f"    {linha}")
        res = predict(vetor)
        reconhecido = nomes[res.index(max(res))]
        correto = "OK" if reconhecido == vogal_ref else "ATENCAO"
        print(f"  Saida: {[round(x,2) for x in res]}"
              f"  -> reconhecido: {reconhecido}  [{correto}]")
        print()
