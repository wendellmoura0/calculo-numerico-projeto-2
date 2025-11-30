# Projeto de Cálculo Numérico - Unidade 2 (2025.2)
**Universidade Federal do Vale do São Francisco (UNIVASF)** *Engenharia de Computação*

## 🚀 Sobre o Projeto
Este projeto consiste no desenvolvimento de uma aplicação web interativa para resolução de problemas reais de engenharia utilizando métodos numéricos. O sistema foi desenvolvido em **Python** utilizando o framework **Streamlit**, permitindo a entrada dinâmica de dados e visualização de resultados em tempo real.

🔗 **Acesse a aplicação online:** [CLIQUE AQUI PARA ACESSAR O SISTEMA](https://calculo-numerico-projeto-2-awdbbakyzxzet7yn7puvr7.streamlit.app/)

---

## 🛠️ Funcionalidades Implementadas

O sistema resolve quatro problemas distintos sorteados para a equipe:

### 1. Planejamento de Produção (Sistemas Lineares - Métodos Diretos)
* **Problema:** Otimização de recursos (metal, plástico, borracha) para produção de componentes.
* **Método:** Resolução de sistema $Ax=b$ via bibliotecas otimizadas (Decomposição LU implícita).
* **Destaque:** Arredondamento lógico para números inteiros (peças físicas).

### 2. Ponte de Wheatstone (Sistemas Lineares - Métodos Iterativos)
* **Problema:** Determinação de correntes em circuito elétrico complexo desbalanceado.
* **Método:** Gauss-Seidel.
* **Destaque:** Análise de convergência e gráfico de decaimento do erro.

### 3. Ajuste de Curvas (Mínimos Quadrados)
* **Problema:** Análise de dados experimentais.
* **Método:** Regressão Linear, Polinomial (Parábola) e Exponencial (Linearizada).
* **Destaque:** Comparação automática do Erro Quadrático Total para sugerir o melhor modelo.

### 4. Hidrologia (Integração Numérica)
* **Problema:** Cálculo da área da seção transversal de um rio.
* **Método:** Regra dos Trapézios e Regra de Simpson 1/3 (Repetidas).
* **Destaque:** Validação automática da paridade de subintervalos para o método de Simpson.

---

## 💻 Tecnologias Utilizadas
* **Linguagem:** Python 3.x
* **Interface:** Streamlit
* **Cálculo Numérico:** NumPy, SciPy
* **Visualização:** Matplotlib, Pandas

## 👥 Autores
* Caio Vinícius Soares Rosa de Souza
* Lucas Gomes de Lucena
* Wendell Moura Leite

---

## ⚙️ Como rodar localmente

1. Clone o repositório:
   ```bash
   git clone [https://github.com/wendellmoura0/calculo-numerico-projeto-2.git](https://github.com/wendellmoura0/calculo-numerico-projeto-2.git)
