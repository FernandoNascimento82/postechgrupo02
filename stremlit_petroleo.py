import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
# Importe outras bibliotecas necessárias

def load_data():
    # Função para carregar dados do preço do petróleo
    data = pd.read_csv('petroleo_data_preparado.csv')
    return data

def main():
    st.title('Dashboard de Preços do Petróleo')

    data = load_data()

    # Exibindo os dados
    st.write(data)
    
    data_corte = '2023-01-01'
    train_data = data[data['DATA'] < data_corte]
    test_data = data[data['DATA'] >= data_corte]
    
    modelo_arima = ARIMA(train_data['VALOR'], order=(2, 1, 3))  # Substituindo p=2, d=1, q=3
    modelo_arima_fit = modelo_arima.fit()
    previsoes = modelo_arima_fit.forecast(steps=len(test_data))
    
    grfc = plt.figure(figsize=(10, 6))
    plt.plot(test_data['DATA'], test_data['VALOR'], label='Valores Reais')
    plt.plot(test_data['DATA'], previsoes, color='red', label='Previsões')
    plt.title('Previsão de Preço do Petróleo')
    plt.xlabel('Data')
    plt.ylabel('Preço')
    plt.legend()
        
    st.pyplot(grfc) 
    
    p = d = q = range(0, 3)  # Definindo intervalos para p, d, q
    pdq = list(itertools.product(p, d, q))  # Todas as combinações de p, d, q

    menor_aic = float('inf')
    melhor_pdq = None

    for combinacao in pdq:
        try:
            modelo_temp = ARIMA(train_data['VALOR'], order=combinacao)
            resultado_temp = modelo_temp.fit()
            if resultado_temp.aic < menor_aic:
                menor_aic = resultado_temp.aic
                melhor_pdq = combinacao
        except:
            continue
        st.write('*Identificado melhor modelo p d q:* ',melhor_pdq)
        
    st.write('**Melhor modelo p d q:** ',melhor_pdq)

    modelo_arima_otimizado = ARIMA(data['VALOR'], order=(melhor_pdq))
    modelo_arima_otimizado_fit = modelo_arima_otimizado.fit()
    
    previsoes_futuras = modelo_arima_otimizado_fit.forecast(steps=30)
    
    data_final = data['DATA'].iloc[-1]
    datas_futuras = pd.date_range(start=data_final, periods=31, inclusive='right')  # 30 dias após a última data

    datas_futuras = np.datetime_as_string(datas_futuras, unit='D')
    
    df = data[(data['DATA'] >="2024-01-01")]

    grfc2 = plt.figure(figsize=(10, 6))
    plt.plot(df['DATA'], df['VALOR'], label='Dados Históricos')
    plt.plot(datas_futuras, previsoes_futuras, color='red', label='Previsões Futuras')
    plt.title('Previsões Futuras de Preço do Petróleo')
    plt.xlabel('Data')
    plt.ylabel('Preço')
    plt.xticks(rotation=90)
    plt.legend()
    
    st.pyplot(grfc2) 
    st.write(datas_futuras.dtype)
    st.write(datas_futuras)
    st.write(df['DATA'].dtype)
    


    # Gráfico do preço do petróleo
    #st.line_chart(teste)

    # Aqui, você pode adicionar mais funcionalidades, gráficos e previsões

if __name__ == "__main__":
    main()
