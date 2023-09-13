import numpy as np  # Importa a biblioteca numpy para manipulação de matrizes.

# Define as matrizes A, B e C como matrizes NumPy.
A = np.array([1, 0, 1])
B = np.array([1, 1, 0])
C = np.array([0, 0, 1])

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
                # Loop para iterar(percorrer) sobre as matrizes em 'matrizes'.
                for matriz in matrizes:
                    # Atualiza o valor em 'pesos[i][j]' somando o produto dos neurônios 'i' e 'j' da matriz atual.
                    pesos[i][j] += matriz[i] * matriz[j]

    return pesos  # Retorna a matriz de pesos calculada.

# Cria uma lista com as matrizes A, B e C.
matrizes = [A, B, C]

# Calcula os pesos da Rede Neural Hopfield chamando a função 'calcular_pesos_hopfield' e armazena em 'pesos_hopfield'.
pesos_hopfield = calcular_pesos_hopfield(matrizes)

# Exibe os pesos da Rede Neural Hopfield.
print("Pesos da Rede Neural Hopfield:")
print(pesos_hopfield)
