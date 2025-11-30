import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sympy import sympify, lambdify
from sympy.abc import x
from matplotlib import rcParams

rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 10


st.set_page_config(
    page_title="Ferramentas de Cálculo Numérico",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

def resolver_sistema_linear(A, b, interpretar_producao=True):
    """Resolve um sistema de equações lineares Ax = b e exibe o resultado no Streamlit."""
    
    st.subheader("Matriz de Coeficientes A")
    st.dataframe(pd.DataFrame(A))
    st.subheader("Vetor de Termos Independentes b")
    st.dataframe(pd.DataFrame(b, columns=['b']))
    
    try:
        x = np.linalg.solve(A, b)
        
        st.subheader("Solução Matemática (x)")
        df_solucao = pd.DataFrame(x, columns=['Valor'])
        df_solucao.index = [f'x{i+1}' for i in range(len(x))]
        st.dataframe(df_solucao.style.format("{:.6f}"))
        
        if interpretar_producao:
            st.subheader("Interpretação no Contexto de Produção")
            solucao_inteira = np.round(x).astype(int)
            
            st.info(f"Solução Arredondada (x1, x2, x3): {solucao_inteira}")
            
            st.markdown(f"""
            - **Componente 1 (x1):** {solucao_inteira[0]} unidades
            - **Componente 2 (x2):** {solucao_inteira[1]} unidades
            - **Componente 3 (x3):** {solucao_inteira[2]} unidades
            
            **Total de componentes produzidos:** {np.sum(solucao_inteira)} unidades
            """)
            
    except np.linalg.LinAlgError:
        st.error("Erro: A matriz de coeficientes é singular. O sistema pode não ter solução única.")
    except ValueError as e:
        st.error(f"Erro de dimensão: {e}. Certifique-se de que A é uma matriz quadrada e b tem o mesmo número de linhas.")

def page_sistemas_lineares():
    st.title("Resolução de Sistemas Lineares 3x3 (T1, Q2)")
    st.markdown("---")
    
    st.sidebar.title("📋 Navegação")
    page = st.sidebar.radio(
        "Selecione uma seção:",
        ["Exemplo Padrão", "Inserir Novo Sistema 3x3"]
    )
    
    if page == "Exemplo Padrão":
        st.header("Exemplo Padrão (Problema de Produção)")
        st.info("Utilizando o Exemplo Padrão de Produção (Ax = b).")
        
        A_exemplo = np.array([
            [15.0, 17.0, 19.0],
            [0.30, 0.40, 0.55],
            [1.0, 1.2, 1.5]
        ])
        b_exemplo = np.array([3890.0, 95.0, 282.0])
        
        st.markdown("""
        **Sistema:**
        - Metal:     15*x1 + 17*x2 + 19*x3   = 3890
        - Plástico:  0.30*x1 + 0.40*x2 + 0.55*x3 = 95
        - Borracha:  1.0*x1 + 1.2*x2 + 1.5*x3  = 282
        """)
        
        if st.button("Executar Exemplo Padrão", key="exec_sl_exemplo"):
            resolver_sistema_linear(A_exemplo, b_exemplo, interpretar_producao=True)
            
    elif page == "Inserir Novo Sistema 3x3":
        st.header("Inserir Novo Sistema 3x3")
        st.markdown("Insira os coeficientes da Matriz A (3x3) e os termos independentes do Vetor b (3x1).")
        
        st.subheader("Matriz A (3x3)")
        A_input = []
        for i in range(3):
            cols = st.columns(3)
            row = []
            for j in range(3):
                default_val = 1.0 if i == j else 0.0
                val = cols[j].number_input(f"A[{i+1},{j+1}]", value=default_val, key=f"A_{i}_{j}")
                row.append(val)
            A_input.append(row)
            
        st.subheader("Vetor b (3x1)")
        b_input = []
        cols_b = st.columns(3)
        for i in range(3):
            val = cols_b[i].number_input(f"b[{i+1}]", value=1.0, key=f"b_{i}")
            b_input.append(val)
            
        if st.button("Resolver Sistema", key="exec_sl_user"):
            try:
                A_user = np.array(A_input)
                b_user = np.array(b_input)
                resolver_sistema_linear(A_user, b_user, interpretar_producao=False)
            except Exception as e:
                st.error(f"Ocorreu um erro: {e}")


def gauss_seidel_detailed(A, b, x0, max_iter, tol):
    """Implementação do método de Gauss-Seidel para o sistema 3x3."""
    n = len(b)
    x = x0.copy()
    history = [x.copy()]
    
    for k in range(1, max_iter + 1):
        x_new = x.copy()
        
        for i in range(n):
            soma = 0
            for j in range(n):
                soma += A[i, j] * x_new[j] if j < i else A[i, j] * x[j]
            
            soma_correta = 0
            for j in range(n):
                if i != j:
                    soma_correta += A[i, j] * x_new[j] if j < i else A[i, j] * x[j]
            
            x_new[i] = (b[i] - soma_correta) / A[i, i]
        
        history.append(x_new.copy())
        
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, history, k
        
        x = x_new
    
    return x, history, max_iter

def calcular_correntes_finais(i1, i2, i3, R1, R2, R3, R4, R5, E):
    """Calcula as correntes finais I1 a I6 a partir das correntes de malha i1, i2, i3."""
    
    I_R1 = i1
    I_R2 = i2
    I_R3 = i2 - i3
    I_R4 = i1 - i3
    I_R5 = i1 - i2
    
    I_total = I_R1 + I_R4
    
    return {
        "I_R1": I_R1,
        "I_R2": I_R2,
        "I_R3": I_R3,
        "I_R4": I_R4,
        "I_R5": I_R5,
        "I_total": I_total
    }

def page_ponte_wheatstone():
    st.title("Ponte de Wheatstone - Método de Gauss-Seidel (T2, Q1)")
    st.markdown("---")
    
    st.sidebar.title("📋 Navegação")
    page = st.sidebar.radio(
        "Selecione uma seção:",
        ["📚 Teoria", "🔧 Calculadora", "📊 Resultados Detalhados"]
    )
    
    if page == "📚 Teoria":
        st.header("📚 Fundamentação Teórica")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "Ponte de Wheatstone",
            "Leis de Kirchhoff",
            "Sistema de Equações",
            "Método de Gauss-Seidel"
        ])
        
        with tab1:
            st.subheader("O que é a Ponte de Wheatstone?")
            st.write("""
            A **Ponte de Wheatstone** é um circuito elétrico clássico utilizado para medir resistências desconhecidas 
            com grande precisão. O circuito é composto por quatro resistores dispostos em uma configuração de ponte, 
            com uma fonte de tensão e um galvanômetro (ou voltímetro) no centro.
            """)
            
            st.subheader("Diagrama do Circuito")
            st.code("""
                        ┌─────────────────┐
                        │                 │
                        E                 A
                        │                 │
                        │         I₁      │
                        │        ╱╲       │
                        │       ╱  ╲      │
                        │      ╱ R₁ ╲     │
                        │     ╱      ╲    │
                        │    ╱        ╲   │
                        B──────────────────C
                        │         I₅      │
                        │        ╱╲       │
                        │       ╱  ╲      │
                        │      ╱ R₅ ╲     │
                        │     ╱      ╲    │
                        │    ╱        ╲   │
                        │   ╱          ╲  │
                        │  ╱            ╲ │
                        │ ╱              ╲│
                        D                 │
                        │                 │
                        └─────────────────┘
            """)
        
        with tab2:
            st.subheader("Leis de Kirchhoff")
            st.write("""
            **1️⃣ Lei dos Nós:** A soma das correntes que entram em um nó é igual à soma das correntes que saem.
            $$\\sum I_{entrada} = \\sum I_{saída}$$
            
            **2️⃣ Lei das Malhas:** A soma das diferenças de potencial (tensões) em qualquer malha fechada é igual a zero.
            $$\\sum V = 0$$
            """)
        
        with tab3:
            st.subheader("Derivação do Sistema de Equações Lineares")
            st.write("""
            Utilizando o **Método das Correntes de Malha** com as correntes $i_1, i_2, i_3$:
            
            **Malha 1 (Esquerda):** $(R_1 + R_4 + R_5)i_1 - R_5 i_2 - R_4 i_3 = E$
            **Malha 2 (Direita):** $-R_5 i_1 + (R_2 + R_3 + R_5)i_2 - R_3 i_3 = 0$
            **Malha 3 (Externa):** $-R_4 i_1 - R_3 i_2 + (R_3 + R_4)i_3 = 0$
            
            **Forma Matricial $A\\mathbf{x} = \\mathbf{b}$:**
            $$
            \\begin{pmatrix}
            R_1 + R_4 + R_5 & -R_5 & -R_4 \\\\
            -R_5 & R_2 + R_3 + R_5 & -R_3 \\\\
            -R_4 & -R_3 & R_3 + R_4
            \\end{pmatrix}
            \\begin{pmatrix}
            i_1 \\\\
            i_2 \\\\
            i_3
            \\end{pmatrix}
            =
            \\begin{pmatrix}
            E \\\\
            0 \\\\
            0
            \\end{pmatrix}
            $$
            """)
        
        with tab4:
            st.subheader("Método Iterativo de Gauss-Seidel")
            st.write("""
            O **Método de Gauss-Seidel** é um algoritmo iterativo que resolve o sistema isolando cada variável:
            
            $$i_1 = \\frac{E + R_5 i_2 + R_4 i_3}{R_1 + R_4 + R_5}$$
            $$i_2 = \\frac{R_5 i_1 + R_3 i_3}{R_2 + R_3 + R_5}$$
            $$i_3 = \\frac{R_4 i_1 + R_3 i_2}{R_3 + R_4}$$
            
            O processo itera até que a diferença entre as soluções consecutivas seja menor que a tolerância $\\epsilon$.
            """)

    elif page == "🔧 Calculadora":
        st.header("🔧 Calculadora - Ponte de Wheatstone")
        
        E_default = 30.0
        R_default = 120.0
        R1_default = 20.0
        
        col_e, col_r1 = st.columns(2)
        E = col_e.number_input("Tensão da Fonte (E) em Volts:", value=E_default, min_value=1.0, format="%.2f", key="E_gs")
        R1 = col_r1.number_input("Resistência R₁ (Ω):", value=R1_default, min_value=1.0, format="%.2f", key="R1_gs")
        
        st.subheader("Resistores R2, R3, R4, R5")
        col_r2, col_r3, col_r4, col_r5 = st.columns(4)
        R2 = col_r2.number_input("R2 (Ω)", value=R_default, min_value=1.0, format="%.2f", key="R2_gs")
        R3 = col_r3.number_input("R3 (Ω)", value=R_default, min_value=1.0, format="%.2f", key="R3_gs")
        R4 = col_r4.number_input("R4 (Ω)", value=R_default, min_value=1.0, format="%.2f", key="R4_gs")
        R5 = col_r5.number_input("R5 (Ω)", value=R_default, min_value=1.0, format="%.2f", key="R5_gs")
        
        st.header("Parâmetros do Método Gauss-Seidel")
        col_max, col_tol = st.columns(2)
        max_iter = col_max.number_input("Máximo de Iterações", value=50, min_value=10, step=1, key="max_iter_gs")
        tol = col_tol.select_slider(
            "Tolerância (ε):",
            options=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
            value=1e-6, key="tol_gs"
        )
        
        st.markdown("---")
        
        if st.button("🚀 Resolver Sistema", use_container_width=True, key="exec_gs"):
            A = np.array([
                [R1 + R4 + R5, -R5, -R4],
                [-R5, R2 + R3 + R5, -R3],
                [-R4, -R3, R3 + R4]
            ], dtype=float)
            b = np.array([E, 0.0, 0.0], dtype=float)
            x0 = np.array([0.0, 0.0, 0.0])
            
            try:
                solucao, historico, num_iter = gauss_seidel_detailed(A, b, x0, max_iter, tol)
                
                st.session_state.gs_solucao = solucao
                st.session_state.gs_historico = historico
                st.session_state.gs_num_iter = num_iter
                st.session_state.gs_A = A
                st.session_state.gs_b = b
                st.session_state.gs_E = E
                st.session_state.gs_R1 = R1
                st.session_state.gs_R2 = R2
                st.session_state.gs_R3 = R3
                st.session_state.gs_R4 = R4
                st.session_state.gs_R5 = R5
                st.session_state.gs_tol = tol
                
                st.success(f"✅ Sistema resolvido em {num_iter} iterações!")
                
                st.subheader("📊 Resultados Resumidos")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Corrente de Malha i₁", f"{solucao[0]:.6f} A")
                with col2:
                    st.metric("Corrente de Malha i₂", f"{solucao[1]:.6f} A")
                with col3:
                    st.metric("Corrente de Malha i₃", f"{solucao[2]:.6f} A")
                
                st.info(f"O método convergiu em **{num_iter}** iterações com tolerância **{tol}**")
                
            except Exception as e:
                st.error(f"Ocorreu um erro durante a execução do Gauss-Seidel: {e}")

    elif page == "📊 Resultados Detalhados":
        st.header("📊 Resultados Detalhados")
        
        if "gs_solucao" not in st.session_state:
            st.warning("⚠️ Primeiro, resolva o sistema na seção 'Calculadora'.")
        else:
            solucao = st.session_state.gs_solucao
            historico = st.session_state.gs_historico
            num_iter = st.session_state.gs_num_iter
            A = st.session_state.gs_A
            b = st.session_state.gs_b
            E = st.session_state.gs_E
            R1 = st.session_state.gs_R1
            R2 = st.session_state.gs_R2
            R3 = st.session_state.gs_R3
            R4 = st.session_state.gs_R4
            R5 = st.session_state.gs_R5
            tol = st.session_state.gs_tol
            
            tab1, tab2, tab3, tab4 = st.tabs([
                "Iterações Detalhadas",
                "Convergência",
                "Correntes do Ramo",
                "Verificação"
            ])
            
            with tab1:
                st.subheader("Detalhamento de Cada Iteração")
                
                df_historico = pd.DataFrame(
                    historico,
                    columns=["i₁ (A)", "i₂ (A)", "i₃ (A)"]
                )
                df_historico.index.name = "Iteração"
                
                erros = []
                for i in range(1, len(historico)):
                    erro = np.linalg.norm(np.array(historico[i]) - np.array(historico[i-1]), ord=np.inf)
                    erros.append(erro)
                
                df_historico["Erro Máximo"] = [0.0] + erros
                
                st.dataframe(df_historico.style.format("{:.6f}"), use_container_width=True)
                
                csv = df_historico.to_csv().encode('utf-8')
                st.download_button(
                    label="📥 Baixar Tabela (CSV)",
                    data=csv,
                    file_name="iteracoes_gauss_seidel.csv",
                    mime="text/csv"
                )
            
            with tab2:
                st.subheader("Análise de Convergência")
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                
                iteracoes = range(len(historico))
                i1_vals = [h[0] for h in historico]
                i2_vals = [h[1] for h in historico]
                i3_vals = [h[2] for h in historico]
                
                ax1.plot(iteracoes, i1_vals, 'o-', label='i₁', linewidth=2, markersize=4)
                ax1.plot(iteracoes, i2_vals, 's-', label='i₂', linewidth=2, markersize=4)
                ax1.plot(iteracoes, i3_vals, '^-', label='i₃', linewidth=2, markersize=4)
                ax1.set_xlabel("Iteração", fontsize=11)
                ax1.set_ylabel("Corrente (A)", fontsize=11)
                ax1.set_title("Convergência das Correntes de Malha", fontsize=12, fontweight='bold')
                ax1.legend(fontsize=10)
                ax1.grid(True, alpha=0.3)
                
                if erros:
                    ax2.semilogy(range(1, len(erros) + 1), erros, 'ro-', linewidth=2, markersize=6)
                    ax2.set_xlabel("Iteração", fontsize=11)
                    ax2.set_ylabel("Erro Máximo (escala log)", fontsize=11)
                    ax2.set_title("Redução do Erro por Iteração", fontsize=12, fontweight='bold')
                    ax2.grid(True, alpha=0.3, which='both')
                else:
                    ax2.text(0.5, 0.5, "Não há iterações suficientes para o gráfico de erro.", ha='center', va='center', transform=ax2.transAxes)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Número de Iterações", num_iter)
                with col2:
                    st.metric("Tolerância (ε)", f"{tol}")
                with col3:
                    if erros:
                        st.metric("Erro Final", f"{erros[-1]:.2e}")
                    else:
                        st.metric("Erro Final", "N/A")
            
            with tab3:
                st.subheader("Correntes em Cada Ramo do Circuito")
                
                correntes_ramo = calcular_correntes_finais(solucao[0], solucao[1], solucao[2], R1, R2, R3, R4, R5, E)
                
                df_correntes = pd.DataFrame({
                    "Ramo": ["R₁", "R₂", "R₃", "R₄", "R₅", "Total (Fonte)"],
                    "Resistência (Ω)": [R1, R2, R3, R4, R5, "-"],
                    "Corrente (A)": [
                        correntes_ramo["I_R1"], 
                        correntes_ramo["I_R2"], 
                        correntes_ramo["I_R3"], 
                        correntes_ramo["I_R4"], 
                        correntes_ramo["I_R5"], 
                        correntes_ramo["I_total"]
                    ],
                    "Queda de Tensão (V)": [
                        correntes_ramo["I_R1"]*R1, 
                        correntes_ramo["I_R2"]*R2, 
                        correntes_ramo["I_R3"]*R3, 
                        correntes_ramo["I_R4"]*R4, 
                        correntes_ramo["I_R5"]*R5, 
                        E
                    ]
                })
                
                st.dataframe(df_correntes.style.format({
                    "Corrente (A)": "{:.6f}",
                    "Queda de Tensão (V)": "{:.6f}"
                }), use_container_width=True)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ramos = ["I(R₁)", "I(R₂)", "I(R₃)", "I(R₄)", "I(R₅)", "I(Total)"]
                correntes = [
                    correntes_ramo["I_R1"], 
                    correntes_ramo["I_R2"], 
                    correntes_ramo["I_R3"], 
                    correntes_ramo["I_R4"], 
                    correntes_ramo["I_R5"], 
                    correntes_ramo["I_total"]
                ]

                cores = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c', '#9467bd', '#8c564b']
                barras = ax.bar(ramos, correntes, color=cores, alpha=0.7, edgecolor='black', linewidth=1.5)

                for bar in barras:
                    yval = bar.get_height()
                    ax.text(
                    bar.get_x() + bar.get_width()/2, 
                    yval + 0.01 * max(correntes), 
                    f'{yval:.6f} A',             
                    ha='center', 
                    va='bottom', 
                    fontsize=9, 
                    fontweight='bold'
                )
                
                ax.set_ylabel("Corrente (A)", fontsize=11)
                ax.set_title("Correntes em Cada Ramo do Circuito", fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                st.pyplot(fig)
            
            with tab4:
                st.subheader("Verificação da Solução")
                
                Ax = A @ solucao
                
                st.markdown("**Sistema Original:**")
                st.code(f"A = {A.tolist()}\nb = {b.tolist()}")
                
                st.markdown("**Solução Encontrada:**")
                st.code(f"x = {solucao.tolist()}")
                
                st.markdown("**Verificação (Ax):**")
                st.code(f"Ax = {Ax.tolist()}")
                
                st.markdown("**Erro Residual (|Ax - b|):**")
                erro_residual = np.abs(Ax - b)
                st.code(f"Erro = {erro_residual.tolist()}")
                
                if np.max(erro_residual) < 1e-4:
                    st.success("✅ Solução verificada com sucesso! O erro residual é muito pequeno.")
                else:
                    st.warning(f"⚠️ Verifique a solução. O erro residual máximo é {np.max(erro_residual):.2e}.")

def erro_quadratico(Y_observado, Y_ajustado):
    """Calcula o erro quadrático cometido."""
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
    
    A = np.array([[N, sum_x], [sum_x, sum_x2]])
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
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(X, Y, 'o', label='Dados Originais F(x)', color='black')
    
    X_plot = np.linspace(X.min(), X.max(), 100)
    
    for res in resultados:
        ajuste = res["Ajuste"]
        coefs = res["Coeficientes"]
        
        if ajuste == "Reta":
            Y_plot = coefs[0] + coefs[1] * X_plot
            ax.plot(X_plot, Y_plot, '-', label=f'Reta (Erro: {res["Erro Quadrático"]:.3f})')
        
        elif ajuste == "Parábola":
            Y_plot = coefs[0] + coefs[1] * X_plot + coefs[2] * (X_plot**2)
            ax.plot(X_plot, Y_plot, '--', label=f'Parábola (Erro: {res["Erro Quadrático"]:.3f})')
            
        elif ajuste == "Exponencial":
            a = math.exp(coefs[0])
            b = coefs[1]
            Y_plot = a * np.exp(b * X_plot)
            ax.plot(X_plot, Y_plot, ':', label=f'Exponencial (Erro: {res["Erro Quadrático"]:.3f})')
            
    ax.set_title('Regressão por Mínimos Quadrados: Comparação de Ajustes')
    ax.set_xlabel('X')
    ax.set_ylabel('F(x)')
    ax.legend()
    ax.grid(True)
    
    return fig

def processar_regressao(X, Y):
    """Executa as 3 regressões e apresenta os resultados."""
    
    resultados = []
    st.subheader("Dados Fornecidos")
    st.dataframe(pd.DataFrame({'X': X, 'F(x)': Y}))
    
    try:
        coefs_linear, Y_ajustado_linear = regressao_linear(X, Y)
        erro_linear = erro_quadratico(Y, Y_ajustado_linear)
        equacao_linear = formatar_polinomio(coefs_linear, "Reta")
        
        resultados.append({
            "Ajuste": "Reta",
            "Equação": equacao_linear,
            "Erro Quadrático": erro_linear,
            "Coeficientes": coefs_linear
        })
    except Exception as e:
        st.error(f"Erro na Regressão Linear: {e}")
        
    try:
        if len(X) < 3:
            st.warning("Aviso: Mínimo de 3 pontos necessários para regressão quadrática. Ignorando.")
        else:
            coefs_quadratica, Y_ajustado_quadratica = regressao_quadratica(X, Y)
            erro_quadratica = erro_quadratico(Y, Y_ajustado_quadratica)
            equacao_quadratica = formatar_polinomio(coefs_quadratica, "Parábola")
            
            resultados.append({
                "Ajuste": "Parábola",
                "Equação": equacao_quadratica,
                "Erro Quadrático": erro_quadratica,
                "Coeficientes": coefs_quadratica
            })
    except Exception as e:
        st.error(f"Erro na Regressão Quadrática: {e}")
        
    try:
        coefs_exp_lin, Y_ajustado_exp = regressao_exponencial(X, Y)
        erro_exp = erro_quadratico(Y, Y_ajustado_exp)
        equacao_exp = formatar_polinomio(coefs_exp_lin, "Exponencial")
        
        resultados.append({
            "Ajuste": "Exponencial",
            "Equação": equacao_exp,
            "Erro Quadrático": erro_exp,
            "Coeficientes": coefs_exp_lin
        })
    except ValueError as e:
        st.warning(f"Aviso na Regressão Exponencial: {e}")
    except Exception as e:
        st.error(f"Erro na Regressão Exponencial: {e}")
        
    if resultados:
        st.subheader("Resumo dos Ajustes")
        df_resultados = pd.DataFrame(resultados)
        df_resultados = df_resultados.sort_values(by="Erro Quadrático")
        st.dataframe(df_resultados[["Ajuste", "Equação", "Erro Quadrático"]].style.format({"Erro Quadrático": "{:.6f}"}))
        
        melhor_ajuste = df_resultados.iloc[0]
        st.success(f"O **melhor ajuste** (menor Erro Quadrático) é a **{melhor_ajuste['Ajuste']}** com Erro Quadrático de **{melhor_ajuste['Erro Quadrático']:.6f}**.")
        
        fig = plotar_ajustes(X, Y, resultados)
        st.pyplot(fig)
    else:
        st.error("Não foi possível realizar nenhum ajuste.")

def page_regressao_minimos_quadrados():
    st.title("Regressão por Mínimos Quadrados (T3, Q3)")
    st.markdown("---")
    
    st.sidebar.title("📋 Navegação")
    page = st.sidebar.radio(
        "Selecione uma seção:",
        ["Exemplo Padrão", "Inserir Novos Dados"]
    )
    
    if page == "Exemplo Padrão":
        st.header("Exemplo Padrão de Regressão")
        X_exemplo = np.array([0.0, 1.5, 2.6, 4.2, 6.0, 8.2, 10.0, 11.4])
        Y_exemplo = np.array([18.0, 13.0, 11.0, 9.0, 6.0, 4.0, 2.0, 1.0])
        
        st.info("Utilizando o Exemplo Padrão.")
        if st.button("Executar Exemplo Padrão", key="exec_regressao_exemplo"):
            processar_regressao(X_exemplo, Y_exemplo)
            
    elif page == "Inserir Novos Dados":
        st.header("Inserir Novos Dados")
        st.markdown("Insira os valores de X e F(x) separados por vírgula ou espaço.")
        
        col_x, col_y = st.columns(2)
        
        x_input = col_x.text_area("Valores de X:", "0, 1.5, 2.6, 4.2, 6.0, 8.2, 10.0, 11.4")
        y_input = col_y.text_area("Valores de F(x):", "18.0, 13.0, 11.0, 9.0, 6.0, 4.0, 2.0, 1.0")
        
        if st.button("Executar Regressão", key="exec_regressao_user"):
            try:
                def parse_input(text):
                    text = text.replace(',', ' ').strip()
                    return np.array([float(val) for val in text.split() if val])

                X_user = parse_input(x_input)
                Y_user = parse_input(y_input)
                
                if len(X_user) != len(Y_user):
                    st.error("Erro: O número de valores para X e F(x) deve ser o mesmo.")
                elif len(X_user) < 2:
                    st.error("Erro: São necessários pelo menos 2 pontos para a regressão linear.")
                else:
                    processar_regressao(X_user, Y_user)
                    
            except ValueError:
                st.error("Erro: Certifique-se de que todos os valores inseridos são números válidos.")
            except Exception as e:
                st.error(f"Ocorreu um erro inesperado: {e}")

def regra_trapezio_repetida_func(X, Y):
    """Calcula a integral usando a Regra do Trapézio Repetida."""
    N = len(X)
    if N < 2:
        return None, "Erro: Mínimo de 2 pontos necessários para a Regra do Trapézio."
    
    h = X[1] - X[0]
    soma_interna = np.sum(Y[1:-1])
    integral = (h / 2) * (Y[0] + 2 * soma_interna + Y[-1])
    return integral, None

def regra_simpson_repetida_func(X, Y):
    """Calcula a integral usando a Regra de Simpson 1/3 Repetida."""
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

def plotar_integracao(X, Y, titulo="Gráfico de Integração Numérica"):
    """Gera o gráfico dos pontos e da área."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(X, Y, '-', label='Função/Perfil', color='blue')
    ax.plot(X, Y, 'o', label='Pontos de Amostragem', color='red')
    ax.fill_between(X, Y, color='skyblue', alpha=0.5, label='Área Calculada')
    
    if "Seção Reta do Rio" in titulo:
        ax.invert_yaxis()
        ax.set_ylabel('Profundidade (m)')
        ax.set_xlabel('Distância a partir da margem esquerda (m)')
    else:
        ax.set_ylabel('f(x)')
        ax.set_xlabel('x')
        
    ax.set_title(titulo)
    ax.legend()
    ax.grid(True)
    
    return fig

def processar_integracao(X, Y, titulo_grafico="Integração Numérica"):
    """Executa as integrações e apresenta os resultados."""
    
    st.subheader("Dados de Amostragem")
    df_dados = pd.DataFrame({'X': X, 'Y (f(x))': Y})
    st.dataframe(df_dados)
    
    N = len(X)
    if N < 2:
        st.error("Erro: Número insuficiente de pontos.")
        return
        
    h = X[1] - X[0]
    st.info(f"Número de pontos (N): {N} | Espaçamento (h): {h:.6f}")
    
    resultados = []
    
    integral_trapezio, erro_trapezio = regra_trapezio_repetida_func(X, Y)
    if erro_trapezio:
        st.warning(f"Trapézio: {erro_trapezio}")
    else:
        resultados.append({"Método": "Trapézio Repetida", "Resultado": integral_trapezio})
        
    integral_simpson, erro_simpson = regra_simpson_repetida_func(X, Y)
    if erro_simpson:
        st.warning(f"Simpson: {erro_simpson}")
    else:
        resultados.append({"Método": "Simpson 1/3 Repetida", "Resultado": integral_simpson})
        
    st.subheader("Resultados da Integração")
    if resultados:
        df_resultados = pd.DataFrame(resultados)
        st.dataframe(df_resultados.style.format({"Resultado": "{:.6f}"}))
    else:
        st.error("Não foi possível calcular a integral com os métodos disponíveis.")
        
    fig = plotar_integracao(X, Y, titulo=titulo_grafico)
    st.pyplot(fig)

def page_integracao_numerica():
    st.title("Integração Numérica (T4, Q1)")
    st.markdown("---")
    
    st.sidebar.title("📋 Navegação")
    page = st.sidebar.radio(
        "Selecione uma seção:",
        ["Exemplo Padrão (Rio)", "Inserir Dados Discretos", "Integrar Função Contínua"]
    )
    
    if page == "Exemplo Padrão (Rio)":
        st.header("Exemplo Padrão (Área da Seção Reta do Rio)")
        st.info("Calculando a Área da Seção Reta do Rio.")
        
        X_exemplo = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
        Y_exemplo = np.array([0.0, 1.8, 4.0, 4.0, 5.0, 6.0, 4.0, 3.6, 3.4, 2.8, 0.0])
        
        if st.button("Executar Exemplo Padrão", key="exec_int_exemplo"):
            processar_integracao(X_exemplo, Y_exemplo, titulo_grafico="Área da Seção Reta do Rio (m²)")
            
    elif page == "Inserir Dados Discretos":
        st.header("Inserir Dados Discretos")
        st.markdown("Insira os valores de X (distância) e Y (f(x) ou profundidade) separados por vírgula ou espaço. **Os valores de X devem ser igualmente espaçados.**")
        
        col_x, col_y = st.columns(2)
        
        x_input = col_x.text_area("Valores de X:", "0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20")
        y_input = col_y.text_area("Valores de Y (f(x)):", "0, 1.8, 4.0, 4.0, 5.0, 6.0, 4.0, 3.6, 3.4, 2.8, 0.0")
        
        if st.button("Executar Integração (Dados Discretos)", key="exec_int_discreto"):
            try:
                def parse_input(text):
                    text = text.replace(',', ' ').strip()
                    return np.array([float(val) for val in text.split() if val])

                X_user = parse_input(x_input)
                Y_user = parse_input(y_input)
                
                if len(X_user) != len(Y_user):
                    st.error("Erro: O número de valores para X e Y deve ser o mesmo.")
                elif len(X_user) < 2:
                    st.error("Erro: São necessários pelo menos 2 pontos.")
                else:
                    h_calc = X_user[1] - X_user[0]
                    if not np.allclose(np.diff(X_user), h_calc):
                        st.error("Erro: Os pontos de X não estão igualmente espaçados.")
                    else:
                        processar_integracao(X_user, Y_user, titulo_grafico="Integração de Dados Discretos")
                        
            except ValueError:
                st.error("Erro: Certifique-se de que todos os valores inseridos são números válidos.")
            except Exception as e:
                st.error(f"Ocorreu um erro inesperado: {e}")
                
    elif page == "Integrar Função Contínua":
        st.header("Integrar Função Contínua")
        st.markdown("Insira a função $f(x)$, os limites de integração $a$ e $b$, e o número de subintervalos $n$.")
        
        func_str = st.text_input("Função f(x) (ex: exp(x), 1/x, x**2):", "x**2")
        
        col_a, col_b, col_n = st.columns(3)
        a = col_a.number_input("Limite Inferior 'a'", value=0.0, format="%.2f")
        b = col_b.number_input("Limite Superior 'b'", value=1.0, format="%.2f")
        n = col_n.number_input("Número de Subintervalos 'n' (deve ser par para Simpson)", value=10, min_value=2, step=2)
        
        if st.button("Executar Integração (Função Contínua)", key="exec_int_funcao"):
            try:
                if b <= a:
                    st.error("Erro: O limite superior 'b' deve ser maior que o limite inferior 'a'.")
                elif n % 2 != 0:
                    st.error("Erro: O número de subintervalos 'n' deve ser par para a Regra de Simpson 1/3.")
                else:
                    X_func = np.linspace(a, b, n + 1)
                    
                    f_expr = sympify(func_str)
                    f_lambda = lambdify(x, f_expr, "numpy")
                    Y_func = f_lambda(X_func)
                    
                    if np.any(np.isinf(Y_func)) or np.any(np.isnan(Y_func)):
                        st.error("Erro: A função resultou em valores inválidos (ex: divisão por zero) no intervalo.")
                    else:
                        titulo = f"Integral de f(x) = {func_str} de {a} a {b} (n={n})"
                        processar_integracao(X_func, Y_func, titulo_grafico=titulo)
                        
            except Exception as e:
                st.error(f"Ocorreu um erro ao processar a função: {e}")
                st.info("Dica: Use 'x' como variável e funções como 'sin(x)', 'cos(x)', 'exp(x)', 'log(x)'.")

def page_home():
    st.title("Bem-vindo ao Kit de Ferramentas de Cálculo Numérico")
    st.markdown("---")
    st.header("Selecione a Questão que Deseja Executar")
    
    st.info("""
    Este aplicativo integra quatro diferentes problemas de cálculo numérico em uma única interface Streamlit.
    Selecione uma das opções abaixo para acessar a ferramenta e seus modos de execução (Exemplo Padrão, Inserção de Dados, Teoria, etc.).
    """)
    
    if 'selected_page' not in st.session_state:
        st.session_state.selected_page = "home"
        
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("1. Resolução de Sistemas Lineares (T1, Q2)", use_container_width=True):
            st.session_state.selected_page = "sistemas_lineares"
            st.rerun()
            
        if st.button("3. Regressão por Mínimos Quadrados (T3, Q3)", use_container_width=True):
            st.session_state.selected_page = "regressao"
            st.rerun()
            
    with col2:
        if st.button("2. Ponte de Wheatstone - Gauss-Seidel (T2, Q1)", use_container_width=True):
            st.session_state.selected_page = "ponte_wheatstone"
            st.rerun()
            
        if st.button("4. Integração Numérica (T4, Q1)", use_container_width=True):
            st.session_state.selected_page = "integracao_numerica"
            st.rerun()

def main_app():
    
    if 'selected_page' not in st.session_state:
        st.session_state.selected_page = "home"
        
    if st.session_state.selected_page != "home":
        if st.sidebar.button("⬅️ Voltar para o Menu Principal"):
            st.session_state.selected_page = "home"
            st.rerun()
            
    pagina = st.session_state.selected_page
    
    if pagina == "home":
        page_home()
        
    elif pagina == "sistemas_lineares":
        page_sistemas_lineares()
        
    elif pagina == "ponte_wheatstone":
        page_ponte_wheatstone()
        
    elif pagina == "regressao":
        page_regressao_minimos_quadrados()
        
    elif pagina == "integracao_numerica":
        page_integracao_numerica()

if __name__ == "__main__":
    main_app()