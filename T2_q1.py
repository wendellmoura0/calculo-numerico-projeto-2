import numpy as np

def gauss_seidel(A, b, x0, max_iter=50, tol=1e-6):
    """
    Para resolver um sistema de equações lineares Ax = b usando o método de Gauss-Seidel.
    
    Args:
        A (np.array): Matriz de coeficientes.
        b (np.array): Vetor de termos independentes.
        x0 (np.array): Chute inicial.
        max_iter (int): Número máximo de iterações.
        tol (float): Tolerância para o critério de parada.
        
    Returns:
        np.array: Solução aproximada.
        list: Histórico das iterações.
    """
    n = len(b)
    x = x0.copy()
    history = [x.tolist()]
    
    print("--- Detalhamento das Iterações de Gauss-Seidel ---")
    print(f"Iteração 0: i1={x[0]:.6f}, i2={x[1]:.6f}, i3={x[2]:.6f}")

    for k in range(1, max_iter + 1):
        x_new = x.copy()
        
        x_new[0] = (b[0] - A[0, 1] * x[1] - A[0, 2] * x[2]) / A[0, 0]
        
        x_new[1] = (b[1] - A[1, 0] * x_new[0] - A[1, 2] * x[2]) / A[1, 1]
        
        x_new[2] = (b[2] - A[2, 0] * x_new[0] - A[2, 1] * x_new[1]) / A[2, 2]
        
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            x = x_new
            history.append(x.tolist())
            print(f"Convergência alcançada na Iteração {k}.")
            break
            
        x = x_new
        history.append(x.tolist())
        print(f"Iteração {k}: i1={x[0]:.6f}, i2={x[1]:.6f}, i3={x[2]:.6f}")
        
    return x, history

A = np.array([
    [260.0, -120.0, -120.0],
    [-120.0, 360.0, -120.0],
    [-120.0, -120.0, 240.0]
])

b = np.array([30.0, 0.0, 0.0])

x0 = np.array([0.0, 0.0, 0.0])

solucao_malha, historico = gauss_seidel(A, b, x0, max_iter=50, tol=1e-6)

print("\n--- Solução Final (Correntes de Malha) ---")
print(f"i1 (Malha Esquerda): {solucao_malha[0]:.6f} A")
print(f"i2 (Malha Direita): {solucao_malha[1]:.6f} A")
print(f"i3 (Malha Externa): {solucao_malha[2]:.6f} A")

i1_final = solucao_malha[0]
i2_final = solucao_malha[1]
i3_final = solucao_malha[2]

I1 = i1_final
I2 = i2_final
I3 = i3_final - i2_final
I4 = i1_final - i3_final
I5 = i1_final - i2_final
I6 = i3_final

print("\n--- Correntes do Ramo (I1 a I6) ---")
print(f"I1 (em R1): {I1:.6f} A")
print(f"I2 (em R2): {I2:.6f} A")
print(f"I3 (em R3): {I3:.6f} A")
print(f"I4 (em R4): {I4:.6f} A")
print(f"I5 (em R5): {I5:.6f} A")
print(f"I6 (Corrente total): {I6:.6f} A")