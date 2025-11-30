import numpy as np
import math
import matplotlib.pyplot as plt


def erro_quadratico(Y_observado, Y_ajustado):
    """
    Calcula o erro quadrático cometido: E = Σ[F(xi) - G(xi)]²
    Onde F(xi) é o valor observado (Y_observado) e G(xi) é o valor ajustado (Y_ajustado).
    """
    return np.sum((Y_observado - Y_ajustado)**2)

def formatar_polinomio(coefs, tipo):
    """Formata os coeficientes em uma string de equação."""
    if tipo == "Reta":
        return f"G(x) = {coefs[0]:.4f} + {coefs[1]:.4f}*x"
    elif tipo == "Parábola":
        return f"G(x) = {coefs[0]:.4f} + {coefs[1]:.4f}*x + {coefs[2]:.4f}*x^2"
    elif tipo == "Exponencial":

        a = math.exp(coefs[0])
        b = coefs[1]
        return f"G(x) = {a:.4f} * e^({b:.4f}*x)"
    return ""


def regressao_linear(X, Y):
    """Ajusta os dados a uma reta: G(x) = a0 + a1*x"""
    
    N = len(X)
    sum_x = np.sum(X)
    sum_y = np.sum(Y)
    sum_x2 = np.sum(X**2)
    sum_xy = np.sum(X * Y)
    
    A = np.array([
        [N, sum_x],
        [sum_x, sum_x2]
    ])
    b = np.array([sum_y, sum_xy])
    
    coefs = np.linalg.solve(A, b)
    Y_ajustado = coefs[0] + coefs[1] * X
    
    return coefs, Y_ajustado

def regressao_quadratica(X, Y):
    """Ajusta os dados a uma parábola: G(x) = a0 + a1*x + a2*x^2"""
    
    N = len(X)
    sum_x = np.sum(X)
    sum_x2 = np.sum(X**2)
    sum_x3 = np.sum(X**3)
    sum_x4 = np.sum(X**4)
    sum_y = np.sum(Y)
    sum_xy = np.sum(X * Y)
    sum_x2y = np.sum((X**2) * Y)
    
    A = np.array([
        [N, sum_x, sum_x2],
        [sum_x, sum_x2, sum_x3],
        [sum_x2, sum_x3, sum_x4]
    ])
    b = np.array([sum_y, sum_xy, sum_x2y])
    
    coefs = np.linalg.solve(A, b)
    Y_ajustado = coefs[0] + coefs[1] * X + coefs[2] * (X**2)
    
    return coefs, Y_ajustado

def regressao_exponencial(X, Y):
    """Ajusta os dados a uma exponencial: G(x) = a*e^(b*x)"""
    
    if np.any(Y <= 0):
        raise ValueError("Regressão exponencial requer que todos os valores de F(x) sejam positivos.")
        
    Y_lin = np.log(Y)
    
    coefs_lin, Y_lin_ajustado = regressao_linear(X, Y_lin)
    
    Y_ajustado = np.exp(Y_lin_ajustado)
    
    return coefs_lin, Y_ajustado


def plotar_ajustes(X, Y, resultados):
    """Gera o gráfico dos dados originais e das curvas de ajuste."""
    
    plt.figure(figsize=(10, 6))
    plt.plot(X, Y, 'o', label='Dados Originais F(x)', color='black')
    
    X_plot = np.linspace(X.min(), X.max(), 100)
    
    for res in resultados:
        ajuste = res["Ajuste"]
        coefs = res["Coeficientes"]
        
        if ajuste == "Reta":
            Y_plot = coefs[0] + coefs[1] * X_plot
            plt.plot(X_plot, Y_plot, '-', label=f'Reta (Erro: {res["Erro Quadrático"]:.3f})')
        
        elif ajuste == "Parábola":
            Y_plot = coefs[0] + coefs[1] * X_plot + coefs[2] * (X_plot**2)
            plt.plot(X_plot, Y_plot, '--', label=f'Parábola (Erro: {res["Erro Quadrático"]:.3f})')
            
        elif ajuste == "Exponencial":
            a = math.exp(coefs[0])
            b = coefs[1]
            Y_plot = a * np.exp(b * X_plot)
            plt.plot(X_plot, Y_plot, ':', label=f'Exponencial (Erro: {res["Erro Quadrático"]:.3f})')
            
    plt.title('Regressão por Mínimos Quadrados: Comparação de Ajustes')
    plt.xlabel('X')
    plt.ylabel('F(x)')
    plt.legend()
    plt.grid(True)
    
    caminho_grafico = "C:/Users/lgluc/Downloads/regressao_ajustes.png"
    plt.savefig(caminho_grafico)
    plt.close()
    
    return caminho_grafico


def processar_regressao(X, Y):
    """Executa as 3 regressões e apresenta os resultados."""
    
    print("\n===================================================================")
    print("       REGRESSÃO POR MÍNIMOS QUADRADOS: RETA, PARÁBOLA E EXPONENCIAL")
    print("===================================================================")
    
    print("\n--- Dados Fornecidos ---")
    print(f"X (Variável Independente): {X}")
    print(f"F(x) (Variável Dependente): {Y}")
    
    resultados = []
    
    print("\n\n--- 1. Regressão Linear (Reta) ---")
    try:
        coefs_linear, Y_ajustado_linear = regressao_linear(X, Y)
        erro_linear = erro_quadratico(Y, Y_ajustado_linear)
        equacao_linear = formatar_polinomio(coefs_linear, "Reta")
        
        print(f"Coeficientes [a0, a1]: {coefs_linear}")
        print(f"Equação de Ajuste: {equacao_linear}")
        print(f"Erro Quadrático (Σ[F(xi) - G(xi)]²): {erro_linear:.6f}")
        
        resultados.append({
            "Ajuste": "Reta",
            "Equação": equacao_linear,
            "Erro Quadrático": erro_linear,
            "Coeficientes": coefs_linear
        })
    except Exception as e:
        print(f"Erro na Regressão Linear: {e}")
        
    print("\n\n--- 2. Regressão Quadrática (Parábola) ---")
    try:
        if len(X) < 3:
            print("Aviso: Mínimo de 3 pontos necessários para regressão quadrática. Ignorando.")
        else:
            coefs_quadratica, Y_ajustado_quadratica = regressao_quadratica(X, Y)
            erro_quadratica = erro_quadratico(Y, Y_ajustado_quadratica)
            equacao_quadratica = formatar_polinomio(coefs_quadratica, "Parábola")
            
            print(f"Coeficientes [a0, a1, a2]: {coefs_quadratica}")
            print(f"Equação de Ajuste: {equacao_quadratica}")
            print(f"Erro Quadrático (Σ[F(xi) - G(xi)]²): {erro_quadratica:.6f}")
            
            resultados.append({
                "Ajuste": "Parábola",
                "Equação": equacao_quadratica,
                "Erro Quadrático": erro_quadratica,
                "Coeficientes": coefs_quadratica
            })
    except Exception as e:
        print(f"Erro na Regressão Quadrática: {e}")
        
    print("\n\n--- 3. Regressão Exponencial ---")
    try:
        coefs_exp_lin, Y_ajustado_exp = regressao_exponencial(X, Y)
        erro_exp = erro_quadratico(Y, Y_ajustado_exp)
        equacao_exp = formatar_polinomio(coefs_exp_lin, "Exponencial")
        
        print(f"Coeficientes Lineares [ln(a), b]: {coefs_exp_lin}")
        print(f"Equação de Ajuste: {equacao_exp}")
        print(f"Erro Quadrático (Σ[F(xi) - G(xi)]²): {erro_exp:.6f}")
        
        resultados.append({
            "Ajuste": "Exponencial",
            "Equação": equacao_exp,
            "Erro Quadrático": erro_exp,
            "Coeficientes": coefs_exp_lin
        })
    except ValueError as e:
        print(f"Erro na Regressão Exponencial: {e}")
    except Exception as e:
        print(f"Erro na Regressão Exponencial: {e}")
        
    print("\n\n===================================================================")
    print("                         RESUMO DOS AJUSTES                        ")
    print("===================================================================")
    
    print("{:<15} {:<50} {:<20}".format("Ajuste", "Equação G(x)", "Erro Quadrático"))
    print("-" * 85)
    for res in resultados:
        print("{:<15} {:<50} {:<20.6f}".format(res["Ajuste"], res["Equação"], res["Erro Quadrático"]))
    print("-" * 85)
    
    if resultados:
        melhor_ajuste = min(resultados, key=lambda x: x["Erro Quadrático"])
        print(f"\nO melhor ajuste (menor Erro Quadrático) é a {melhor_ajuste['Ajuste']}.")
        
        caminho_grafico = plotar_ajustes(X, Y, resultados)
        print(f"\nGráfico gerado e salvo em: {caminho_grafico}")
        return caminho_grafico
    else:
        print("\nNão foi possível realizar nenhum ajuste.")
        return None


def exemplo_1():
    """Roda o problema original de regressão (Exemplo 1)."""
    print("\n===================================================================")
    print("                  EXEMPLO 1: PROBLEMA ORIGINAL                     ")
    print("===================================================================")
    
    X = np.array([0.0, 1.5, 2.6, 4.2, 6.0, 8.2, 10.0, 11.4])
    Y = np.array([18.0, 13.0, 11.0, 9.0, 6.0, 4.0, 2.0, 1.0])
    
    return processar_regressao(X, Y)

def input_dados_usuario():
    """Permite ao usuário inserir os dados para a regressão."""
    print("\n===================================================================")
    print("                  MODO INTERATIVO: NOVOS DADOS                     ")
    print("===================================================================")
    
    while True:
        try:
            x_str = input("Insira os valores de X (separados por espaço): ")
            X = np.array([float(x) for x in x_str.split()])
            
            y_str = input("Insira os valores de F(x) (separados por espaço): ")
            Y = np.array([float(y) for y in y_str.split()])
            
            if len(X) != len(Y):
                print("Erro: O número de valores para X e F(x) deve ser o mesmo.")
                continue
            
            if len(X) < 2:
                print("Erro: São necessários pelo menos 2 pontos para a regressão linear.")
                continue
            
            return processar_regressao(X, Y)
            
        except ValueError:
            print("Erro: Insira apenas números válidos.")
        except Exception as e:
            print(f"Ocorreu um erro: {e}")
            return None

def main():
    """Função principal para iniciar o programa e apresentar as opções."""
    
    print("\n===================================================================")
    print("        FERRAMENTA DE REGRESSÃO POR MÍNIMOS QUADRADOS        ")
    print("===================================================================")
    
    caminho_grafico = None
    
    while True:
        print("\nEscolha uma opção:")
        print("1 - Rodar o Exemplo 1 (Dados Originais)")
        print("2 - Inserir novos dados (Modo Interativo)")
        print("3 - Sair")
        
        escolha = input("Sua escolha (1, 2 ou 3): ")
        
        if escolha == '1':
            caminho_grafico = exemplo_1()
        elif escolha == '2':
            caminho_grafico = input_dados_usuario()
        elif escolha == '3':
            print("Programa encerrado. Obrigado!")
            break
        else:
            print("Opção inválida. Por favor, escolha 1, 2 ou 3.")
            
    if caminho_grafico:
        print(f"\nO gráfico da última execução foi salvo em: {caminho_grafico}")

if __name__ == "__main__":
    try:
        import numpy as np
        import matplotlib.pyplot as plt
    except ImportError:
        print("As bibliotecas 'numpy' e 'matplotlib' são necessárias. Instale com: pip install numpy matplotlib")
        exit()
        
    main()