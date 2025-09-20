import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

class FastForecast:
    def __init__(self):
        self.encoders = {}
        self.model = None
        self.feature_columns = []

    def load_and_process_data(self):
        BASE_DIR = os.path.dirname(os.path.dirname(__file__)) 
        data_path = os.path.join(BASE_DIR, "data")
        files = glob.glob(os.path.join(data_path, "*.parquet"))
        
        # Load data
        df_pdv = pd.read_parquet(files[0])
        df_transactions = pd.read_parquet(files[1])
        df_products = pd.read_parquet(files[2])
        
        print(f"Carregados: PDV {df_pdv.shape}, Transações {df_transactions.shape}, Produtos {df_products.shape}")
        
        # Sample data for faster processing (10% sample)
        df_transactions = df_transactions.sample(frac=0.1, random_state=42)
        print(f"Sample das transações: {df_transactions.shape}")
        
        # Merge data
        df_main = df_transactions.merge(df_products, left_on='internal_product_id', right_on='produto', how='inner')
        df_main = df_main.merge(df_pdv, left_on='internal_store_id', right_on='pdv', how='inner')
        
        print(f"Dataset combinado: {df_main.shape}")
        
        # Weekly aggregation
        df_main['transaction_date'] = pd.to_datetime(df_main['transaction_date'])
        df_main = df_main[df_main['transaction_date'] >= '2022-01-01']
        
        df_main['year'] = df_main['transaction_date'].dt.year
        df_main['week'] = df_main['transaction_date'].dt.isocalendar().week
        
        weekly_agg = df_main.groupby(['internal_store_id', 'internal_product_id', 'year', 'week']).agg({
            'quantity': 'sum',
            'net_value': 'sum'
        }).reset_index()
        
        weekly_agg.columns = ['pdv', 'produto', 'year', 'week', 'quantidade_total', 'valor_liquido_total']
        
        # Add categorical info
        cat_info = df_main[['internal_product_id', 'categoria', 'premise']].drop_duplicates()
        weekly_agg = weekly_agg.merge(cat_info, left_on='produto', right_on='internal_product_id', how='left')
        
        print(f"Dados semanais: {weekly_agg.shape}")
        
        return weekly_agg

    def create_features(self, df):
        df = df.copy()
        df = df.sort_values(['pdv', 'produto', 'year', 'week']).reset_index(drop=True)
        
        # Basic features
        df['month'] = df['week'].apply(lambda x: (x-1)//4 + 1)  # Approximate month
        df['quarter'] = df['month'].apply(lambda x: (x-1)//3 + 1)
        
        # Encode categories
        for col in ['categoria', 'premise']:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str).fillna('unknown'))
                self.encoders[col] = le
        
        # Lag features (simplified)
        df['quantidade_lag_1'] = df.groupby(['pdv', 'produto'])['quantidade_total'].shift(1)
        df['quantidade_lag_2'] = df.groupby(['pdv', 'produto'])['quantidade_total'].shift(2)
        
        # Rolling mean
        df['quantidade_rolling_mean_4'] = df.groupby(['pdv', 'produto'])['quantidade_total'].transform(
            lambda x: x.rolling(4, min_periods=1).mean()
        )
        
        return df

    def train_model(self, df):
        feature_cols = ['year', 'week', 'month', 'quarter', 'categoria_encoded', 'premise_encoded',
                       'quantidade_lag_1', 'quantidade_lag_2', 'quantidade_rolling_mean_4']
        
        df_clean = df.dropna(subset=['quantidade_total', 'quantidade_lag_1'])
        
        X = df_clean[feature_cols].fillna(0)
        y = df_clean['quantidade_total']
        
        train_size = int(len(df_clean) * 0.8)
        X_train, X_val = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_val = y.iloc[:train_size], y.iloc[train_size:]
        
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbose': -1,
            'random_state': 42
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)
        
        self.model = lgb.train(params, train_data, valid_sets=[val_data], num_boost_round=100)
        self.feature_columns = feature_cols
        
        return df_clean

    def generate_predictions(self, df_processed):
        # Get top 1000 most active combinations
        combinations = df_processed.groupby(['pdv', 'produto'])['quantidade_total'].agg(['count', 'sum']).reset_index()
        combinations = combinations[(combinations['count'] >= 4) & (combinations['sum'] > 5)]
        combinations = combinations.nlargest(1000, 'sum')
        
        predictions = []
        
        for week in range(1, 5):
            for _, combo in combinations.iterrows():
                pdv, produto = combo['pdv'], combo['produto']
                
                hist = df_processed[(df_processed['pdv'] == pdv) & (df_processed['produto'] == produto)]
                if len(hist) < 2:
                    continue
                
                last_record = hist.iloc[-1]
                
                pred_row = {
                    'year': 2023,
                    'week': week,
                    'month': 1,
                    'quarter': 1,
                    'categoria_encoded': last_record.get('categoria_encoded', 0),
                    'premise_encoded': last_record.get('premise_encoded', 0),
                    'quantidade_lag_1': hist['quantidade_total'].iloc[-1] if len(hist) >= 1 else 0,
                    'quantidade_lag_2': hist['quantidade_total'].iloc[-2] if len(hist) >= 2 else 0,
                    'quantidade_rolling_mean_4': hist['quantidade_total'].tail(4).mean()
                }
                
                X_pred = pd.DataFrame([pred_row])[self.feature_columns].fillna(0)
                pred_qty = max(0, int(round(self.model.predict(X_pred)[0])))
                
                predictions.append({
                    'semana': week,
                    'pdv': int(pdv),
                    'produto': int(produto),
                    'quantidade': pred_qty
                })
        
        return pd.DataFrame(predictions)

    def run_pipeline(self):
        df = self.load_and_process_data()
        df_features = self.create_features(df)
        df_processed = self.train_model(df_features)
        predictions = self.generate_predictions(df_processed)
        
        # Save results
        output_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "previsoes_janeiro_2023_fast.csv")
        predictions.to_csv(output_file, index=False, sep=';')
        accuracy = mean_absolute_error(df_processed['quantidade_total'], self.model.predict(df_processed[self.feature_columns].fillna(0)))
        print(f"Previsões salvas: {output_file}")
        print(f"Total: {len(predictions)} previsões")
        print(f"Precisão: {accuracy}")
        return predictions

if __name__ == "__main__":
    model = FastForecast()
    results = model.run_pipeline()
    print("Concluído!")
    print(results.head())
