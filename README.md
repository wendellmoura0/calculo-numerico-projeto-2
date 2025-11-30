# Projeto II - Cálculo Numérico (UNIVASF)

![Status](https://img.shields.io/badge/STATUS-CONCLUÍDO-brightgreen) ![Python](https://img.shields.io/badge/PYTHON-3.10+-blue) ![Streamlit](https://img.shields.io/badge/FRAMEWORK-STREAMLIT-red)

Projeto desenvolvido para a disciplina de **Cálculo Numérico** da **Universidade Federal do Vale do São Francisco (UNIVASF)**, sob orientação do **Prof. Jorge Luis Cavalcanti Ramos**.

## 🔗 Acesso à Aplicação
A ferramenta está implantada e disponível para uso online. Clique no botão abaixo para acessar:

[![Acessar Simulador](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://calculo-numerico-projeto-2-awdbbakyzxzet7yn7puvr7.streamlit.app/)

---

## 📝 Descrição do Projeto

Este projeto consiste em uma **Aplicação Web (SaaS)** desenvolvida em Python, criada para solucionar problemas reais de engenharia através de métodos numéricos. O objetivo é demonstrar a aplicação prática de algoritmos para resolução de sistemas lineares, ajuste de dados experimentais e cálculo de áreas irregulares.

A interface foi construída para permitir a **entrada dinâmica de dados**, possibilitando que o usuário simule diferentes cenários de produção, circuitos elétricos e hidrologia, visualizando os resultados matemáticos e gráficos em tempo real.

## 👨‍💻 Equipe

* **Caio Vinícius Soares Rosa de Souza**
* **Lucas Gomes de Lucena**
* **Wendell Moura Leite**

---

## 🚀 Funcionalidades Principais

O sistema é dividido em quatro módulos computacionais, conforme os problemas propostos:

### 1. Planejamento de Produção (Métodos Diretos)
* **Contexto:** Otimização de linha de produção com recursos limitados (Metal, Plástico, Borracha).
* **Solução:** Modelagem matricial $Ax=b$ e resolução via Decomposição LU.
* **Saída:** Quantidade exata de componentes a serem produzidos (com tratamento de arredondamento inteiro).

### 2. Análise de Circuitos (Métodos Iterativos)
* **Contexto:** Cálculo de correntes em uma Ponte de Wheatstone desbalanceada.
* **Solução:** Aplicação do **Método de Gauss-Seidel**.
* **Destaque:** Visualização da convergência do erro a cada iteração e verificação de matriz diagonal dominante.

### 3. Regressão de Dados (Mínimos Quadrados)
* **Contexto:** Ajuste de curvas para dados experimentais.
* **Solução:** Comparação automática entre modelos **Linear, Polinomial (Quadrático) e Exponencial**.
* **Destaque:** Cálculo do Erro Quadrático Total para sugerir matematicamente o melhor modelo ao engenheiro.

### 4. Hidrologia (Integração Numérica)
* **Contexto:** Determinação da área da seção transversal de um rio baseada em sondagens de profundidade.
* **Solução:** Implementação comparativa das **Regras dos Trapézios** e **Simpson 1/3**.
* **Destaque:** Geração gráfica do perfil do leito do rio e validação automática de paridade de intervalos.

---

## 🛠 Tecnologias Utilizadas

* **Linguagem:** Python
* **Interface Gráfica:** Streamlit
* **Computação Científica:** NumPy & SciPy
* **Visualização de Dados:** Matplotlib & Pandas
