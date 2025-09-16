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

    def advanced_feature_engineering(self, df):
        df = df.copy()
        
        # Conversão de datas
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        df['reference_date'] = pd.to_datetime(df['reference_date'])
        
        # Features temporais
        df['year'] = df['transaction_date'].dt.year
        df['month'] = df['transaction_date'].dt.month
        df['day'] = df['transaction_date'].dt.day
        df['dayofweek'] = df['transaction_date'].dt.dayofweek
        df['quarter'] = df['transaction_date'].dt.quarter
        df['week'] = df['transaction_date'].dt.isocalendar().week
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        df['is_month_end'] = df['transaction_date'].dt.is_month_end.astype(int)
        df['is_month_start'] = df['transaction_date'].dt.is_month_start.astype(int)
        
        # Features de sazonalidade
        df['season'] = df['month'].map({12: 'Winter', 1: 'Winter', 2: 'Winter',
                                       3: 'Spring', 4: 'Spring', 5: 'Spring',
                                       6: 'Summer', 7: 'Summer', 8: 'Summer',
                                       9: 'Fall', 10: 'Fall', 11: 'Fall'})
        
        # Features financeiras
        df['profit_margin'] = df['gross_profit'] / df['gross_value']
        df['discount_rate'] = df['discount'] / df['gross_value']
        df['tax_rate'] = df['taxes'] / df['gross_value']
        df['price_per_unit'] = df['net_value'] / df['quantity']
        df['value_category'] = pd.cut(df['net_value'], bins=5, labels=['Low', 'Low-Med', 'Medium', 'Med-High', 'High'])
        
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
        df = df.sort_values(['internal_store_id', 'internal_product_id', 'transaction_date'])
        
        return df

    

    def run_complete_forecast_pipeline(self):
        df_pdv, df_transactions, df_products = self.import_and_merge_data()
        df_combined = self.data_merge(df_pdv, df_transactions, df_products)

        df_features = self.advanced_feature_engineering(df_combined)
        print(df_features.head())

if __name__ == "__main__":

    forecast_model = TrainForecast()
    results, predictions = forecast_model.run_complete_forecast_pipeline()
    print(results, predictions)
