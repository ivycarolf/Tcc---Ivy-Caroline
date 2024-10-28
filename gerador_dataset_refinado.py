import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Gerar dados fictícios para o exemplo
np.random.seed(42)

# Número de exemplos no dataset
n = 1000

# Criar colunas de variáveis explicativas
tipos_obra = np.random.choice(['Residencial', 'Comercial', 'Industrial'], n)
localizacoes = np.random.choice(['Urbana', 'Rural'], n)
area_construida = np.random.uniform(50, 5000, n)  # Área em metros quadrados
complexidade = np.random.choice(['Baixa', 'Média', 'Alta'], n)
n_pavimentos = np.random.randint(1, 10, n)  # Número de pavimentos
clima = np.random.choice(['Estável', 'Variável'], n)
equipe_tamanho = np.random.randint(5, 100, n)

# Criar colunas de variáveis dependentes (custo e prazo)
custo_total = area_construida * np.random.uniform(1000, 3000) + n_pavimentos * 10000
prazo_estimado = (area_construida / 100) + n_pavimentos * 2 + np.random.uniform(1, 6, n)

# Montar DataFrame
df = pd.DataFrame({
    'Tipo_Obra': tipos_obra,
    'Localizacao': localizacoes,
    'Area_Construida_m2': area_construida,
    'Complexidade': complexidade,
    'Numero_Pavimentos': n_pavimentos,
    'Clima': clima,
    'Equipe_Tamanho': equipe_tamanho,
    'Custo_Total_R$': custo_total,
    'Prazo_Estimado_Meses': prazo_estimado
})

# Salvar o dataset refinado
df.to_csv('dataset_projetos_civis_refinado.csv', index=False)