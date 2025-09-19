# Solução do grupo Clairvoyance para o Hackaton Forecast Big Data

## Modelo de Previsão
- **Algoritmo**: LightGBM (Gradient Boosting otimizado para grandes datasets)
- **Features**: Lag features, médias móveis, sazonalidade, features financeiras
- **Output**: CSV com previsões para as 4 primeiras semanas de Janeiro 2023

## Para rodar o projeto
```bash
# Instalar dependências
pip install -r requirements.txt

# Executar modelo de previsão
python .\public\src\train_forecast.py
```

## Arquivos de saída
- `previsoes_janeiro_2023.csv`: Contém as previsões com colunas semana, pdv, produto, quantidade

## Otimizações implementadas
- Processamento em chunks para lidar com 6M+ registros
- Filtros para combinações PDV-Produto mais ativas
- Early stopping e validação temporal
- Encoding eficiente de variáveis categóricas
