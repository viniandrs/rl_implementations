import numpy as np
import torch
import matplotlib.pyplot as plt

a = 'red'

def plot_arrows_from_qtable(q_table):
    """"
    Cria um gráfico do tabuleiro em que cada posição possui setas proporcionais ao QValor de cada ação.
    """
    board = np.ones((4, 12))
    board[0, 1:11] = 0
    plt.imshow(board, cmap='gray', origin='lower')
    plt.grid()

    for position in range(48):
        if position in range(37, 47):
            continue

        y = 3 - position // 12 #+ 0.5 # divisao inteira
        x = position % 12 #+ 0.5 # resto da divisao

        # normalizando para o intervalo [0, 0.5]
        T = 0.0001
        exp= np.exp((q_table[position] - np.max(q_table[position]))/T)
        normalized_q_values = 0.4 * (exp / np.sum(exp))
        
        # cima
        plt.arrow(x, y, 0, normalized_q_values[0], head_width=0.1, head_length=0.1, fc=a, ec=a)
        # direita
        plt.arrow(x, y, normalized_q_values[1], 0, head_width=0.1, head_length=0.1, fc=a, ec=a)
        # baixo
        plt.arrow(x, y, 0, -normalized_q_values[2], head_width=0.1, head_length=0.1, fc=a, ec=a)
        # esquerda
        plt.arrow(x, y, -normalized_q_values[3], 0, head_width=0.1, head_length=0.1, fc=a, ec=a)

    # pintando o buraco de preto
    plt.xticks(np.arange(13) - 0.5)
    plt.yticks(np.arange(5) - 0.5)
    plt.axis('scaled')
    plt.show()
    plt.savefig('arrowplot_cliffwalking.png')

def plot_arrows_from_qnet(q_net):
    """"
    Cria um gráfico do tabuleiro em que cada posição possui setas proporcionais ao Q-valor de cada ação.
    """
    board = np.ones((4, 12))
    board[0, 1:11] = 0
    plt.imshow(board, cmap='gray', origin='lower')
    plt.grid()

    for position in range(48):
        if position in range(37, 47):
            continue
        
        x, y = position % 12, 3-position // 12

        qvals = q_net(position)
        T = 0.0001
        normalized_q_values = 0.4 * torch.softmax(qvals/T, dim=-1).detach().cpu().numpy()

        # cima
        plt.arrow(x, y, 0, normalized_q_values[0], head_width=0.1, head_length=0.1, fc=a, ec=a)
        # direita
        plt.arrow(x, y, normalized_q_values[1], 0, head_width=0.1, head_length=0.1, fc=a, ec=a)
        # baixo
        plt.arrow(x, y, 0, -normalized_q_values[2], head_width=0.1, head_length=0.1, fc=a, ec=a)
        # esquerda
        plt.arrow(x, y, -normalized_q_values[3], 0, head_width=0.1, head_length=0.1, fc=a, ec=a)

    # pintando o buraco de preto
    plt.xticks(np.arange(13) - 0.5)
    plt.yticks(np.arange(5) - 0.5)
    plt.axis('scaled')
    plt.savefig('arrowplot_cliffwalking.png')
    plt.show()
