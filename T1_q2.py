import numpy as np

def resolver_sistema_linear(A, b, interpretar_producao=True):
    """
    Resolve um sistema de equações lineares Ax = b usando métodos diretos
    (inversão de matriz via numpy.linalg.solve).

    Args:
        A (np.array): Matriz de coeficientes.
        b (np.array): Vetor de termos independentes.
        interpretar_producao (bool): Se True, exibe a interpretação para o contexto de produção.

    Returns:
        None
    """
    print("\n--- Item (b): Programa para Resolução (Método Direto - numpy.linalg.solve) ---")
    print("A função 'resolver_sistema_linear' utiliza o método direto (Decomposição LU) do NumPy.")
    
    try:
        x = np.linalg.solve(A, b)
        
        print("\n--- Item (c): Solução do Sistema Linear ---")
        print("Matriz de Coeficientes A:")
        print(A)
        print("\nVetor de Termos Independentes b:")
        print(b)
        
        print(f"\nSolução Matemática (x): {x}")
        
        if interpretar_producao:
            solucao_inteira = np.round(x).astype(int)
            
            print(f"Solução Arredondada (x1, x2, x3): {solucao_inteira}")
            print("\nInterpretação (Quantidades de Componentes):")
            print(f"Componente 1 (x1): {solucao_inteira[0]} unidades")
            print(f"Componente 2 (x2): {solucao_inteira[1]} unidades")
            print(f"Componente 3 (x3): {solucao_inteira[2]} unidades")
            print(f"\nTotal de componentes produzidos: {np.sum(solucao_inteira)} unidades")
        
    except np.linalg.LinAlgError:
        print("\nErro: A matriz de coeficientes é singular. O sistema pode não ter solução única.")
    except ValueError as e:
        print(f"\nErro de dimensão: {e}. Certifique-se de que A é uma matriz quadrada e b tem o mesmo número de linhas.")

def exemplo_1():
    """
    Resolve o problema original de produção de componentes (Exemplo 1).
    """
    print("\n===================================================================")
    print("                  EXEMPLO 1: PROBLEMA ORIGINAL                     ")
    print("===================================================================")
    
    A = np.array([
        [15.0, 17.0, 19.0],    
        [0.30, 0.40, 0.55],   
        [1.0, 1.2, 1.5]        
    ])
    

    b = np.array([3890.0, 95.0, 282.0])
    
    print("\n--- Item (a): Apresentação do Problema ---")
    print("Definindo as variáveis:")
    print("  x1 = quantidade de Componente 1")
    print("  x2 = quantidade de Componente 2")
    print("  x3 = quantidade de Componente 3")
    
    print("\nConvertendo os totais disponíveis de kg para g:")
    print("  Metal:     3,89 kg  -> 3890 g")
    print("  Plástico:  0,095 kg -> 95 g")
    print("  Borracha:  0,282 kg -> 282 g")
    
    print("\nO problema é descrito pelo seguinte sistema de equações lineares (Ax = b):")
    print("\nEquação do Metal:     15*x1 + 17*x2 + 19*x3   = 3890")
    print("Equação do Plástico:  0.30*x1 + 0.40*x2 + 0.55*x3 = 95")
    print("Equação da Borracha:  1.0*x1 + 1.2*x2 + 1.5*x3  = 282")
    
    resolver_sistema_linear(A, b, interpretar_producao=True)

def input_dados_usuario():
    """
    Permite ao usuário inserir os dados para um novo sistema linear 3x3.
    """
    print("\n===================================================================")
    print("                  MODO INTERATIVO: NOVOS DADOS                     ")
    print("===================================================================")
    
    print("\nInsira os dados para um sistema de equações lineares 3x3 (Ax = b).")
    print("A matriz A é de 3x3 e o vetor b é de 3x1.")
    
    A_list = []
    for i in range(3):
        while True:
            try:
                linha = input(f"Insira os 3 coeficientes da linha {i+1} da Matriz A (separados por espaço): ")
                coeficientes = [float(x) for x in linha.split()]
                if len(coeficientes) == 3:
                    A_list.append(coeficientes)
                    break
                else:
                    print("Erro: Você deve inserir exatamente 3 números.")
            except ValueError:
                print("Erro: Insira apenas números válidos.")
    
    b_list = []
    while True:
        try:
            termos = input("Insira os 3 termos independentes do Vetor b (separados por espaço): ")
            termos_independentes = [float(x) for x in termos.split()]
            if len(termos_independentes) == 3:
                b_list = termos_independentes
                break
            else:
                print("Erro: Você deve inserir exatamente 3 números.")
        except ValueError:
            print("Erro: Insira apenas números válidos.")
            
    A = np.array(A_list)
    b = np.array(b_list)
    
    resolver_sistema_linear(A, b, interpretar_producao=False)
    
def main():
    """
    Função principal para iniciar o programa e apresentar as opções.
    """
    print("\n===================================================================")
    print("        RESOLUÇÃO DE SISTEMAS LINEARES 3x3 (MÉTODOS DIRETOS)        ")
    print("===================================================================")
    
    while True:
        print("\nEscolha uma opção:")
        print("1 - Rodar o Exemplo 1 (Problema de Produção Original)")
        print("2 - Inserir novos dados (Modo Interativo)")
        print("3 - Sair")
        
        escolha = input("Sua escolha (1, 2 ou 3): ")
        
        if escolha == '1':
            exemplo_1()
        elif escolha == '2':
            input_dados_usuario()
        elif escolha == '3':
            print("Programa encerrado. Obrigado!")
            break
        else:
            print("Opção inválida. Por favor, escolha 1, 2 ou 3.")

if __name__ == "__main__":
    try:
        import numpy as np
    except ImportError:
        print("A biblioteca 'numpy' é necessária. Instale com: pip install numpy")
        exit()
        
    main() 