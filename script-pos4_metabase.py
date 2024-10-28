import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import seaborn as sns
import statsmodels.api as sm

import matplotlib.pyplot as plt

from sqlalchemy import create_engine

# Definir se os graficos serão plotados
plot = 0

# Defina a conexão com o banco de dados MySQL
engine = create_engine('mysql+pymysql://ivycaroline:Ivyecaio#7@localhost/dataset_banco')

# Função para calcular o MAPE
def calcular_mape(y_true, y_pred):
    # Garantir que não haja divisão por zero
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Função para calcular o MSE
def calcular_mse(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2)

# Carregar o dataset refinado
df = pd.read_csv('dataset_projetos_civis_refinado.csv')

# Separar variáveis explicativas (X) e variáveis alvo (y) para orçamento e prazo
X = df[['Tipo_Obra', 'Localizacao', 'Area_Construida_m2', 'Complexidade', 'Numero_Pavimentos', 'Clima', 'Equipe_Tamanho']]
y_custo = df['Custo_Total_R$']
y_prazo = df['Prazo_Estimado_Meses']

# Dividir o dataset em treino e teste
X_train, X_test, y_custo_train, y_custo_test, y_prazo_train, y_prazo_test = train_test_split(
    X, y_custo, y_prazo, test_size=0.2, random_state=42)

# Definir os pré-processamentos para variáveis categóricas e numéricas
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Area_Construida_m2', 'Numero_Pavimentos', 'Equipe_Tamanho']),
        ('cat', OneHotEncoder(), ['Tipo_Obra', 'Localizacao', 'Complexidade', 'Clima'])
    ])

# Criar pipeline para o modelo de Custo usando Random Forest
modelo_custo = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Treinar o modelo de custo
modelo_custo.fit(X_train, y_custo_train)

# Prever o custo no conjunto de teste
y_custo_pred = modelo_custo.predict(X_test)

# Avaliar o desempenho do modelo de custo
## mse_custo = mean_squared_error(y_custo_test, y_custo_pred)
mse_custo = calcular_mse(y_custo_test, y_custo_pred)
mae_custo = mean_absolute_error(y_custo_test, y_custo_pred)
print(f'MAE de custo: {mae_custo:.2f}')
print(f'MSE de custo: {mse_custo:.2f}')

rmse = np.sqrt(mse_custo)
print(f'RMSE de custo: {rmse:.2f}')

r2_custo = r2_score(y_custo_test, y_custo_pred)

# Calcular MAPE
mape = calcular_mape(y_custo_test, y_custo_pred)
print(f'MAPE de custo: {mape:.2f}%')

print(f'R² de custo: {r2_custo}')

# Criar pipeline para o modelo de Prazo usando Random Forest
modelo_prazo = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Treinar o modelo de prazo
modelo_prazo.fit(X_train, y_prazo_train)

# Prever o prazo no conjunto de teste
y_prazo_pred = modelo_prazo.predict(X_test)

# Avaliar o desempenho do modelo de prazo
## mse_prazo = mean_squared_error(y_prazo_test, y_prazo_pred)
mae_prazo = mean_absolute_error(y_prazo_test, y_prazo_pred)
print(f'MAE de prazo: {mae_prazo:.2f}')

mse_prazo = calcular_mse(y_prazo_test, y_prazo_pred)
print(f'MSE de prazo: {mse_prazo:.2f}')

rmse = np.sqrt(mse_prazo)
print(f'RMSE de prazo: {rmse:.2f}')

r2_prazo = r2_score(y_prazo_test, y_prazo_pred)

# Calcular MAPE
mape = calcular_mape(y_prazo_test, y_prazo_pred)
print(f'MAPE de prazo: {mape:.2f}%')

print(f'R² de prazo: {r2_prazo}')

if plot == 1:
    # Plotando os resultados de custo
    plt.scatter(y_custo_test, y_custo_pred)
    plt.xlabel('Custo Real')
    plt.ylabel('Custo Previsto')
    plt.title('Custo Real vs Custo Previsto')
    z = np.polyfit(y_custo_test, y_custo_pred, 1)
    p = np.poly1d(z)
    plt.plot(y_custo_test,p(y_custo_test),"r--")
    plt.show()

    # Plotando os resultados de prazo
    plt.scatter(y_prazo_test, y_prazo_pred)
    plt.xlabel('Prazo Real (meses)')
    plt.ylabel('Prazo Previsto (meses)')
    plt.title('Prazo Real vs Prazo Previsto')
    z = np.polyfit(y_prazo_test, y_prazo_pred, 1)
    p = np.poly1d(z)
    plt.plot(y_prazo_test,p(y_prazo_test),"r--")
    plt.show()

# Exemplo de como usar o modelo para fazer predições com novos dados
novos_dados = pd.DataFrame({
    'Tipo_Obra': ['Residencial'],
    'Localizacao': ['Urbana'],
    'Area_Construida_m2': [200],
    'Complexidade': ['Média'],
    'Numero_Pavimentos': [3],
    'Clima': ['Estável'],
    'Equipe_Tamanho': [50]
})

# Prever Custo e Prazo para novos dados
custo_previsto = modelo_custo.predict(novos_dados)
prazo_previsto = modelo_prazo.predict(novos_dados)

print(f'Custo Previsto: R${custo_previsto[0]:,.2f}')
print(f'Prazo Previsto: {prazo_previsto[0]:.2f} meses')


# Calculando os resíduos de custo
residuos_custo = np.array(y_custo_test) - np.array(y_custo_pred)

if plot == 1:
    # 1. Gráfico de Resíduos vs. Valores Preditos
    plt.figure(figsize=(10, 6))
    plt.scatter(y_custo_pred, residuos_custo)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Resíduos de custo vs Valores Preditos')
    plt.xlabel('Valores Preditos')
    plt.ylabel('Resíduos de custo')
    plt.show()

    # 2. Histograma dos Resíduos
    plt.figure(figsize=(10, 6))
    sns.histplot(residuos_custo, kde=True, bins=30)
    plt.title('Histograma dos Resíduos de custo')
    plt.xlabel('Resíduo de custo')
    plt.ylabel('Frequência')
    plt.show()

    # 3. QQ-Plot (Gráfico Quantil-Quantil) para verificar normalidade dos resíduos
    plt.figure(figsize=(10, 6))
    sm.qqplot(residuos_custo, line='s')
    plt.title('QQ-Plot dos Resíduos de custo')
    plt.show()

# Calculando os resíduos de Prazo
residuos_prazo = np.array(y_prazo_test) - np.array(y_prazo_pred)

if plot == 1:
    # 1. Gráfico de Resíduos vs. Valores Preditos
    plt.figure(figsize=(10, 6))
    plt.scatter(y_prazo_pred, residuos_prazo)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Resíduos de prazo vs Valores Preditos')
    plt.xlabel('Valores Preditos')
    plt.ylabel('Resíduos de prazo')
    plt.show()

    # 2. Histograma dos Resíduos
    plt.figure(figsize=(10, 6))
    sns.histplot(residuos_prazo, kde=True, bins=30)
    plt.title('Histograma dos Resíduos de prazo')
    plt.xlabel('Resíduo de prazo')
    plt.ylabel('Frequência')
    plt.show()

    # 3. QQ-Plot (Gráfico Quantil-Quantil) para verificar normalidade dos resíduos
    plt.figure(figsize=(10, 6))
    sm.qqplot(residuos_prazo, line='s')
    plt.title('QQ-Plot dos Resíduos de prazo')
    plt.show()

# Salvar o DataFrame no banco de dados MySQL
df.to_sql('projetos', con=engine, if_exists='replace', index=False)

# Suponha que y_custo_pred e y_prazo_pred sejam arrays numpy com previsões
# Converta para um DataFrame
df_pred = pd.DataFrame({
    'custo_previsto': y_custo_pred,
    'prazo_previsto': y_prazo_pred
})

df_test = pd.DataFrame({
    'custo_test': y_custo_test,
    'prazo_test': y_prazo_test
})

df_test_pred =pd.DataFrame({
    'custo_test': y_custo_test,
    'prazo_test': y_prazo_test,
    'custo_previsto': y_custo_pred,
    'prazo_previsto': y_prazo_pred
})

# Insira o DataFrame no banco de dados como uma nova tabela
# Substitua 'previsoes_projetos' pelo nome desejado para a tabela
df_pred.to_sql('previsoes_projetos', con=engine, if_exists='replace', index=False)

df_test.to_sql('test_projetos', con=engine, if_exists='replace', index=False)

df_test_pred.to_sql('test_e_previsoes_projetos', con=engine, if_exists='replace', index=False)