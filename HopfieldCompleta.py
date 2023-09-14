import numpy as np

# Define as matrizes A, B e C como matrizes NumPy.
A = np.array([1, -1, 1, 1, -1])
B = np.array([-1, -1, -1, 1, 1])
C = np.array([-1, 1, 1, 1, -1])

# Função para calcular os pesos da Rede Neural Hopfield.
def calcular_pesos_hopfield(matrizes):
    num_neuronios = len(matrizes[0])  # Calcula o número de neurônios com base na primeira matriz em 'matrizes'.
    pesos = np.zeros((num_neuronios, num_neuronios))  # Inicializa a matriz de pesos 'pesos' como uma matriz de zeros.

    # Loop externo para iterar sobre os neurônios (linhas) da matriz de pesos.
    for i in range(num_neuronios):
        # Loop interno para iterar sobre os neurônios (colunas) da matriz de pesos.
        for j in range(num_neuronios):
            # Verifica se o índice 'i' é diferente do índice 'j'.
            if i != j:
                # Loop para iterar (percorrer) sobre as matrizes em 'matrizes'.
                for matriz in matrizes:
                    # Atualiza o valor em 'pesos[i][j]' somando o produto dos neurônios 'i' e 'j' da matriz atual.
                    pesos[i][j] += matriz[i] * matriz[j]

    return pesos  # Retorna a matriz de pesos calculada.

# Função para calcular a saída da Rede Neural Hopfield com base em uma matriz de entrada 'entrada'.
def calcular_saida_hopfield(entrada, pesos):
    num_neuronios = len(entrada)
    saida = np.zeros(num_neuronios)

    for i in range(num_neuronios):
        for j in range(num_neuronios):
            saida[i] += entrada[j] * pesos[i][j]

    return saida

# Função para aplicar a função relé bipolar aos valores da matriz de entrada.
def aplicar_rele_bipolar(matriz):
    return np.where(matriz < 0, -1, 1)

# Função para verificar a similaridade entre duas matrizes.
def calcular_similaridade(matriz1, matriz2):
    return np.sum(matriz1 == matriz2) / len(matriz1)

# Cria uma lista com as matrizes A, B e C.
matrizes = [A, B, C]

# Calcula os pesos da Rede Neural Hopfield chamando a função 'calcular_pesos_hopfield' e armazena em 'pesos_hopfield'.
pesos_hopfield = calcular_pesos_hopfield(matrizes)

# Solicita ao usuário que digite a matriz D.
D = np.array(list(map(int, input("Digite a matriz D (5x5) separada por espaços e linhas: ").split())))

# Realiza o processo até 10 vezes.
for _ in range(10):
    # Calcula a saída da Rede Neural Hopfield com base na matriz D.
    saida = calcular_saida_hopfield(D, pesos_hopfield)

    # Aplica a função relé bipolar à saída.
    D = aplicar_rele_bipolar(saida)

    # Verifica a similaridade entre a matriz D e as amostras iniciais.
    similaridade_A = calcular_similaridade(D, A)
    similaridade_B = calcular_similaridade(D, B)
    similaridade_C = calcular_similaridade(D, C)

    # Verifica se alguma das amostras atingiu a similaridade desejada (95%).
    if similaridade_A >= 0.95:
        print("Matriz D é semelhante à amostra A com", similaridade_A * 100, "% de similaridade.")
        break
    elif similaridade_B >= 0.95:
        print("Matriz D é semelhante à amostra B com", similaridade_B * 100, "% de similaridade.")
        break
    elif similaridade_C >= 0.95:
        print("Matriz D é semelhante à amostra C com", similaridade_C * 100, "% de similaridade.")
        break

# Caso as 10 iterações sejam concluídas sem atingir a similaridade desejada.
print("As 10 iterações foram concluídas. A matriz D final é:")
print(D)
