import os
import glob
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

class TrainForecast:
    def __init__(self):
        self.encoders = {}


    def import_and_merge_data(self):

        BASE_DIR = os.path.dirname(os.path.dirname(__file__)) 
        data_path = os.path.join(BASE_DIR, "data")
        files = glob.glob(os.path.join(data_path, "*.parquet"))
        
        df_pdv = pd.read_parquet(files[0])  # PDV
        df_transactions = pd.read_parquet(files[1])  # Transações
        df_products = pd.read_parquet(files[2])  # Produtos
        
        print(f"Dados carregados:")
        print(f"PDV: {df_pdv.shape}")
        print(f"Transações: {df_transactions.shape}")
        print(f"Produtos: {df_products.shape}")
        
        return df_pdv, df_transactions, df_products

    def data_merge(self, df_pdv, df_transactions, df_products):

        # Merge transações com produtos
        df_main = df_transactions.merge(
            df_products, 
            left_on='internal_product_id', 
            right_on='produto', 
            how='left'
        )
        
        # Merge com dados de PDV
        df_main = df_main.merge(
            df_pdv,
            left_on='internal_store_id',
            right_on='pdv',
            how='left'
        )
        
        print(f"Dataset combinado: {df_main.shape}")
        return df_main

    def weekly_aggregation(self, df):

        df = df.copy()
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        
        # identificador de semana
        df['year'] = df['transaction_date'].dt.year
        df['week'] = df['transaction_date'].dt.isocalendar().week
        df['year_week'] = df['year'].astype(str) + '_' + df['week'].astype(str).str.zfill(2)
        
        # agregação semanal por PDV e produto
        weekly_agg = df.groupby(['internal_store_id', 'internal_product_id', 'year', 'week']).agg({
            'quantity': 'sum',
            'net_value': 'sum',
            'gross_value': 'sum',
            'gross_profit': 'sum',
            'discount': 'sum',
            'taxes': 'sum',
            'transaction_date': 'count'  # Número de transações na semana
        }).reset_index()
        
        # renomeadinha
        weekly_agg.columns = ['pdv', 'produto', 'year', 'week', 'quantidade_total', 
                             'valor_liquido_total', 'valor_bruto_total', 'lucro_bruto_total',
                             'desconto_total', 'impostos_total', 'num_transacoes']
        
        # informações complementares
        product_info = df[['internal_product_id', 'categoria', 'tipos', 'label', 'subcategoria']].drop_duplicates()
        pdv_info = df[['internal_store_id', 'premise', 'categoria_pdv']].drop_duplicates()
        
        weekly_agg = weekly_agg.merge(
            product_info, left_on='produto', right_on='internal_product_id', how='left'
        ).drop('internal_product_id', axis=1)
        
        weekly_agg = weekly_agg.merge(
            pdv_info, left_on='pdv', right_on='internal_store_id', how='left'
        ).drop('internal_store_id', axis=1)

        print(f"Dados agregados semanalmente: {weekly_agg.shape}")
        
        return weekly_agg


    def feature_engineering(self, df):
        df = df.copy()
        
        # conversao de datas
        df['synthetic_date'] = pd.to_datetime(df['year'].astype(str) + df['week'].astype(str) + '1', format='%G%V%u', errors='coerce')
        df['month'] = df['synthetic_date'].dt.month
        df['day'] = df['synthetic_date'].dt.day
        df['dayofweek'] = df['synthetic_date'].dt.dayofweek
        df['quarter'] = df['synthetic_date'].dt.quarter
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        df['is_month_end'] = df['synthetic_date'].dt.is_month_end.astype(int)
        df['is_month_start'] = df['synthetic_date'].dt.is_month_start.astype(int)
        
        # Features de sazonalidade
        df['season'] = df['month'].map({12: 'Winter', 1: 'Winter', 2: 'Winter',
                                       3: 'Spring', 4: 'Spring', 5: 'Spring',
                                       6: 'Summer', 7: 'Summer', 8: 'Summer',
                                       9: 'Fall', 10: 'Fall', 11: 'Fall'})
        
        # Features financeiras
        df['profit_margin'] = df['lucro_bruto_total'] / df['valor_bruto_total']
        df['discount_rate'] = df['desconto_total'] / df['valor_bruto_total']
        df['tax_rate'] = df['impostos_total'] / df['valor_bruto_total']
        df['price_per_unit'] = df['valor_liquido_total'] / df['quantidade_total']
        df['value_category'] = pd.cut(df['valor_liquido_total'], bins=5, labels=['Low', 'Low-Med', 'Medium', 'Med-High', 'High'])
        
        # Encoding de variáveis categóricas
        categorical_cols = ['categoria', 'tipos', 'label', 'subcategoria', 'marca', 
                           'premise', 'categoria_pdv', 'season', 'value_category']
        
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[f'{col}_encoded'] = self.encoders[col].transform(df[col].astype(str))
        
        # Ordena por data para features temporais
        df = df.sort_values(['pdv', 'produto', 'synthetic_date'])

        print(f"Features criadas: {df.shape}")
        
        return df

    def time_series_features(self, df):

        df = df.copy()
        groupby_cols = ['pdv', 'produto']
        
        # ordenando
        df = df.sort_values(['pdv', 'produto', 'year', 'week'])
        
        # semanas anteriores
        for lag in [1, 2, 3, 4, 5, 8, 12, 26, 52]:
            df[f'quantidade_lag_{lag}'] = df.groupby(groupby_cols)['quantidade_total'].shift(lag)
            if lag <= 4:  # para períodos mais próximos
                df[f'valor_lag_{lag}'] = df.groupby(groupby_cols)['valor_liquido_total'].shift(lag)
        
        # médias móveis
        for window in [2, 4, 8, 12, 26]:
            df[f'quantidade_rolling_mean_{window}'] = df.groupby(groupby_cols)['quantidade_total'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df[f'quantidade_rolling_std_{window}'] = df.groupby(groupby_cols)['quantidade_total'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
            if window <= 8:
                df[f'valor_rolling_mean_{window}'] = df.groupby(groupby_cols)['valor_liquido_total'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
        
        # tendência e crescimento
        df['semanas_ativas'] = df.groupby(groupby_cols).cumcount() + 1
        df['quantidade_media_historica'] = df.groupby(groupby_cols)['quantidade_total'].expanding().mean()
        
        # Crescimento semanal
        df['crescimento_semana_anterior'] = df.groupby(groupby_cols)['quantidade_total'].pct_change()
        df['crescimento_4_semanas'] = df.groupby(groupby_cols)['quantidade_total'].pct_change(periods=4)
        
        # sazonais (mesmo período ano anterior)
        df['quantidade_mesmo_periodo_ano_anterior'] = df.groupby(groupby_cols + ['week'])['quantidade_total'].shift(1)
        
        # Volatilidade e estabilidade
        df['cv_quantidade'] = df.groupby(groupby_cols)['quantidade_total'].transform(
            lambda x: x.rolling(window=8, min_periods=2).std() / x.rolling(window=8, min_periods=2).mean()
        )
        
        return df

    
    def run_complete_forecast_pipeline(self):
        df_pdv, df_transactions, df_products = self.import_and_merge_data()
        df_combined = self.data_merge(df_pdv, df_transactions, df_products)

        df_weekly = self.weekly_aggregation(df_combined)

        df_features = self.feature_engineering(df_weekly)
        df_time_features = self.time_series_features(df_features)

        print(df_time_features.head())

if __name__ == "__main__":

    forecast_model = TrainForecast()
    results, predictions = forecast_model.run_complete_forecast_pipeline()
    print(results, predictions)
