import numpy as np
import matplotlib.pyplot as plt
import math
from sympy import sympify, lambdify
from sympy.abc import x

def regra_trapezio_repetida(X, Y):
    """
    Calcula a integral usando a Regra do Trapézio Repetida.
    Assume que os pontos X são igualmente espaçados.
    """
    N = len(X)
    if N < 2:
        return None, "Erro: Mínimo de 2 pontos necessários para a Regra do Trapézio."
    
    h = X[1] - X[0]
    soma_interna = np.sum(Y[1:-1])
    integral = (h / 2) * (Y[0] + 2 * soma_interna + Y[-1])
    return integral, None

def regra_simpson_repetida(X, Y):
    """
    Calcula a integral usando a Regra de Simpson 1/3 Repetida.
    Assume que os pontos X são igualmente espaçados e o número de subintervalos (N-1) é par.
    """
    N = len(X)
    if N < 3:
        return None, "Erro: Mínimo de 3 pontos necessários para a Regra de Simpson."
    
    if (N - 1) % 2 != 0:
        return None, "Erro: O número de subintervalos (N-1) deve ser par para a Regra de Simpson Repetida."
        
    h = X[1] - X[0]
    soma_impares = np.sum(Y[1:-1:2])
    soma_pares = np.sum(Y[2:-1:2])
    integral = (h / 3) * (Y[0] + 4 * soma_impares + 2 * soma_pares + Y[-1])
    return integral, None


def plotar_resultados(X, Y, titulo="Gráfico de Integração Numérica"):
    """Gera o gráfico dos pontos e da área."""
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(X, Y, '-', label='Função/Perfil', color='blue')
    plt.plot(X, Y, 'o', label='Pontos de Amostragem', color='red')
    plt.fill_between(X, Y, color='skyblue', alpha=0.5, label='Área Calculada')
    
    if "Seção Reta do Rio" in titulo:
        plt.gca().invert_yaxis()
        plt.ylabel('Profundidade (m)')
        plt.xlabel('Distância a partir da margem esquerda (m)')
    else:
        plt.ylabel('f(x)')
        plt.xlabel('x')
        
    plt.title(titulo)
    plt.legend()
    plt.grid(True)
    
    caminho_grafico = "C:/Users/lgluc/Downloads/integracao_numerica_resultado.png"
    plt.savefig(caminho_grafico)
    plt.close()
    
    return caminho_grafico

def processar_integracao(X, Y, titulo_grafico="Gráfico de Integração Numérica"):
    """Executa as integrações e apresenta os resultados."""
    
    print(f"\n===================================================================")
    print(f"           INTEGRAÇÃO NUMÉRICA: {titulo_grafico}          ")
    print(f"===================================================================")
    
    print("\n--- Dados de Amostragem ---")
    print(f"X: {np.round(X, 6)}")
    print(f"Y (f(x)): {np.round(Y, 6)}")
    
    N = len(X)
    if N < 2:
        print("Erro: Número insuficiente de pontos.")
        return None
        
    h = X[1] - X[0]
    print(f"Número de pontos (N): {N}")
    print(f"Espaçamento (h): {h:.6f}")
    
    print("\n\n--- 1. Regra do Trapézio Repetida ---")
    integral_trapezio, erro_trapezio = regra_trapezio_repetida(X, Y)
    
    if erro_trapezio:
        print(erro_trapezio)
    else:
        print(f"Resultado (Trapézio): {integral_trapezio:.6f}")
        
    print("\n\n--- 2. Regra de Simpson 1/3 Repetida ---")
    integral_simpson, erro_simpson = regra_simpson_repetida(X, Y)
    
    if erro_simpson:
        print(erro_simpson)
    else:
        print(f"Resultado (Simpson): {integral_simpson:.6f}")
        
    print("\n\n===================================================================")
    print("                         RESUMO DOS RESULTADOS                     ")
    print("===================================================================")
    
    print("{:<25} {:<20}".format("Método", "Resultado"))
    print("-" * 45)
    if integral_trapezio is not None:
        print("{:<25} {:<20.6f}".format("Trapézio Repetida", integral_trapezio))
    if integral_simpson is not None:
        print("{:<25} {:<20.6f}".format("Simpson 1/3 Repetida", integral_simpson))
    print("-" * 45)
    
    caminho_grafico = plotar_resultados(X, Y, titulo=titulo_grafico)
    print(f"\nGráfico gerado e salvo em: {caminho_grafico}")
    return caminho_grafico


def exemplo_1_rio():
    """Roda o problema original de integração numérica (Exemplo 1 - Rio)."""
    print("\n===================================================================")
    print("                  EXEMPLO 1: ÁREA DA SEÇÃO RETA DO RIO             ")
    print("===================================================================")
    
    X = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
    Y = np.array([0.0, 1.8, 4.0, 4.0, 5.0, 6.0, 4.0, 3.6, 3.4, 2.8, 0.0])
    
    return processar_integracao(X, Y, titulo_grafico="Área da Seção Reta do Rio (m²)")

def input_dados_discretos():
    """Permite ao usuário inserir os dados discretos para a integração."""
    print("\n===================================================================")
    print("                  MODO INTERATIVO: DADOS DISCRETOS                 ")
    print("===================================================================")
    
    while True:
        try:
            x_str = input("Insira os valores de X (Distância) separados por espaço (devem ser igualmente espaçados): ")
            X = np.array([float(x) for x in x_str.split()])
            
            y_str = input("Insira os valores de Y (Profundidade/f(x)) separados por espaço: ")
            Y = np.array([float(y) for y in y_str.split()])
            
            if len(X) != len(Y):
                print("Erro: O número de valores para X e Y deve ser o mesmo.")
                continue
            
            if len(X) < 2:
                print("Erro: São necessários pelo menos 2 pontos.")
                continue
                
            h_calc = X[1] - X[0]
            if not np.allclose(np.diff(X), h_calc):
                print("Erro: Os pontos de X não estão igualmente espaçados.")
                continue
            
            return processar_integracao(X, Y, titulo_grafico="Integração de Dados Discretos")
            
        except ValueError:
            print("Erro: Insira apenas números válidos.")
        except Exception as e:
            print(f"Ocorreu um erro: {e}")
            return None

def input_funcao_continua():
    """Permite ao usuário inserir uma função, limites e n para a integração."""
    print("\n===================================================================")
    print("               MODO INTERATIVO: FUNÇÃO CONTÍNUA                    ")
    print("===================================================================")
    
    while True:
        try:
            func_str = input("Insira a função f(x) (ex: 1/x, exp(x), x**2): ")
            
            a = float(input("Limite inferior 'a': "))
            b = float(input("Limite superior 'b': "))
            n = int(input("Número de subintervalos 'n' (deve ser par para Simpson): "))
            
            if n <= 0:
                print("Erro: O número de subintervalos deve ser positivo.")
                continue
            if b <= a:
                print("Erro: O limite superior 'b' deve ser maior que o limite inferior 'a'.")
                continue
                
            h = (b - a) / n
            X = np.linspace(a, b, n + 1)
            
            f_expr = sympify(func_str)
            f_lambda = lambdify(x, f_expr, "numpy")
            Y = f_lambda(X)
            
            if np.any(np.isinf(Y)) or np.any(np.isnan(Y)):
                print("Erro: A função resultou em valores inválidos (ex: divisão por zero) no intervalo.")
                continue
            
            titulo = f"Integral de f(x) = {func_str} de {a} a {b} (n={n})"
            return processar_integracao(X, Y, titulo_grafico=titulo)
            
        except (ValueError, TypeError):
            print("Erro: Verifique se os limites e 'n' são números válidos e se a função está bem formada.")
        except Exception as e:
            print(f"Ocorreu um erro: {e}")
            print("Dica: Use 'x' como variável e funções como 'sin(x)', 'cos(x)', 'exp(x)', 'log(x)'.")
            return None

def main():
    """Função principal para iniciar o programa e apresentar as opções."""
    
    print("\n===================================================================")
    print("        FERRAMENTA DE INTEGRAÇÃO NUMÉRICA (ÁREA DE SEÇÃO RETA)        ")
    print("===================================================================")
    
    caminho_grafico = None
    
    while True:
        print("\nEscolha uma opção:")
        print("1 - Rodar o Exemplo 1 (Área da Seção Reta do Rio)")
        print("2 - Inserir Dados Discretos (Pontos X e Y)")
        print("3 - Resolver Integral de Função Contínua (f(x), a, b, n)")
        print("4 - Sair")
        
        escolha = input("Sua escolha (1, 2, 3 ou 4): ")
        
        if escolha == '1':
            caminho_grafico = exemplo_1_rio()
        elif escolha == '2':
            caminho_grafico = input_dados_discretos()
        elif escolha == '3':
            caminho_grafico = input_funcao_continua()
        elif escolha == '4':
            print("Programa encerrado. Obrigado!")
            break
        else:
            print("Opção inválida. Por favor, escolha 1, 2, 3 ou 4.")
            
    if caminho_grafico:
        print(f"\nO gráfico da última execução foi salvo em: {caminho_grafico}")

if __name__ == "__main__":
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        from sympy import sympify, lambdify
        from sympy.abc import x
    except ImportError:
        print("As bibliotecas 'numpy', 'matplotlib' e 'sympy' são necessárias.")
        print("Instale com: pip install numpy matplotlib sympy")
        exit()
        
    main()

    try:
        import numpy as np
        import matplotlib.pyplot as plt
    except ImportError:
        print("As bibliotecas 'numpy' e 'matplotlib' são necessárias. Instale com: pip install numpy matplotlib")
        exit()
        
    main()