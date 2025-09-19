import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

class TrainForecast:
    def __init__(self):
        self.encoders = {}
        self.chunksize = 50000  # Reduced for better memory management
        self.model = None
        
        self.feature_columns = []

    def import_and_merge_data(self):
        BASE_DIR = os.path.dirname(os.path.dirname(__file__)) 
        data_path = os.path.join(BASE_DIR, "data")
        files = glob.glob(os.path.join(data_path, "*.parquet"))
        
        if len(files) != 3:
            raise ValueError(f"Expected 3 parquet files, found {len(files)}")
        
        # Load with proper column handling
        df_pdv = pd.read_parquet(files[0])
        df_transactions = pd.read_parquet(files[1]) 
        df_products = pd.read_parquet(files[2])
        
        print(f"Dados carregados:")
        print(f"PDV: {df_pdv.shape} - Columns: {df_pdv.columns.tolist()}")
        print(f"Transações: {df_transactions.shape} - Columns: {df_transactions.columns.tolist()}")
        print(f"Produtos: {df_products.shape} - Columns: {df_products.columns.tolist()}")
        
        return df_pdv, df_transactions, df_products

    def data_merge(self, df_pdv, df_transactions, df_products):
        print("Iniciando merge dos dados...")
        
        # Merge transactions with products
        df_main = df_transactions.merge(
            df_products, 
            left_on='internal_product_id', 
            right_on='produto', 
            how='inner'  # Changed to inner to avoid missing data
        )
        
        # Merge with PDV data
        df_main = df_main.merge(
            df_pdv,
            left_on='internal_store_id',
            right_on='pdv',
            how='inner'  # Changed to inner to avoid missing data
        )
        
        print(f"Dataset combinado: {df_main.shape}")
        return df_main

    def weekly_aggregation(self, df):
        df = df.copy()
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        
        # Filter data to ensure we have enough history (2022 data for training)
        df = df[df['transaction_date'] >= '2022-01-01']
        
        # Weekly identifier
        df['year'] = df['transaction_date'].dt.year
        df['week'] = df['transaction_date'].dt.isocalendar().week
        df['year_week'] = df['year'].astype(str) + '_' + df['week'].astype(str).str.zfill(2)
        
        # Weekly aggregation by PDV and product
        weekly_agg = df.groupby(['internal_store_id', 'internal_product_id', 'year', 'week']).agg({
            'quantity': 'sum',
            'net_value': 'sum',
            'gross_value': 'sum', 
            'gross_profit': 'sum',
            'discount': 'sum',
            'taxes': 'sum',
            'transaction_date': 'count'
        }).reset_index()
        
        weekly_agg.columns = ['pdv', 'produto', 'year', 'week', 'quantidade_total', 
                             'valor_liquido_total', 'valor_bruto_total', 'lucro_bruto_total',
                             'desconto_total', 'impostos_total', 'num_transacoes']
        
        # Add complementary information
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
        
        # Date features
        df['synthetic_date'] = pd.to_datetime(df['year'].astype(str) + df['week'].astype(str) + '1', format='%G%V%u', errors='coerce')
        df['month'] = df['synthetic_date'].dt.month
        df['day'] = df['synthetic_date'].dt.day
        df['dayofweek'] = df['synthetic_date'].dt.dayofweek
        df['quarter'] = df['synthetic_date'].dt.quarter
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        df['is_month_end'] = df['synthetic_date'].dt.is_month_end.astype(int)
        df['is_month_start'] = df['synthetic_date'].dt.is_month_start.astype(int)
        
        # Seasonality features
        df['season'] = df['month'].map({12: 0, 1: 0, 2: 0,  # Winter
                                       3: 1, 4: 1, 5: 1,    # Spring
                                       6: 2, 7: 2, 8: 2,    # Summer
                                       9: 3, 10: 3, 11: 3}) # Fall
        
        # Financial features
        df['profit_margin'] = df['lucro_bruto_total'] / (df['valor_bruto_total'] + 1e-8)
        df['discount_rate'] = df['desconto_total'] / (df['valor_bruto_total'] + 1e-8)
        df['tax_rate'] = df['impostos_total'] / (df['valor_bruto_total'] + 1e-8)
        df['price_per_unit'] = df['valor_liquido_total'] / (df['quantidade_total'] + 1e-8)
        
        # Encode categorical variables
        categorical_cols = ['categoria', 'tipos', 'label', 'subcategoria', 'premise', 'categoria_pdv']
        
        for col in categorical_cols:
            if col in df.columns:
                # Clean the values first
                df[col] = df[col].astype(str).fillna('missing')
                
                if col not in self.encoders:
                    # First time - fit the encoder with 'unknown' included
                    self.encoders[col] = LabelEncoder()
                    unique_vals = df[col].unique().tolist()
                    if 'unknown' not in unique_vals:
                        unique_vals.append('unknown')
                    self.encoders[col].fit(unique_vals)
                    df[f'{col}_encoded'] = self.encoders[col].transform(df[col])
                else:
                    # Subsequent chunks - handle unseen categories
                    le = self.encoders[col]
                    known_classes = set(le.classes_)
                    
                    # Map unknown values to 'unknown' if it exists, otherwise to first class
                    if 'unknown' in known_classes:
                        unknown_replacement = 'unknown'
                    else:
                        unknown_replacement = le.classes_[0]
                    
                    vals_mapped = df[col].map(lambda x: x if x in known_classes else unknown_replacement)
                    df[f'{col}_encoded'] = le.transform(vals_mapped)
        
        print(f"Features criadas: {df.shape}")
        return df

    def time_series_features(self, df):
        df = df.copy()
        df = df.sort_values(['pdv', 'produto', 'year', 'week'])
        
        # Reset index to avoid groupby issues
        df = df.reset_index(drop=True)
        
        # Lag features
        for lag in [1, 2, 3, 4, 8, 12]:
            df[f'quantidade_lag_{lag}'] = df.groupby(['pdv', 'produto'])['quantidade_total'].shift(lag)
            if lag <= 4:
                df[f'valor_lag_{lag}'] = df.groupby(['pdv', 'produto'])['valor_liquido_total'].shift(lag)
        
        # Rolling features
        for window in [2, 4, 8, 12]:
            df[f'quantidade_rolling_mean_{window}'] = df.groupby(['pdv', 'produto'])['quantidade_total'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df[f'quantidade_rolling_std_{window}'] = df.groupby(['pdv', 'produto'])['quantidade_total'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
        
        # Trend features - fix expanding window issue
        df['semanas_ativas'] = df.groupby(['pdv', 'produto']).cumcount() + 1
        
        # Use transform instead of expanding for compatibility
        df['quantidade_media_historica'] = df.groupby(['pdv', 'produto'])['quantidade_total'].transform(
            lambda x: x.expanding().mean()
        )
        
        df['crescimento_semana_anterior'] = df.groupby(['pdv', 'produto'])['quantidade_total'].pct_change()
        
        # Additional stable features
        df['quantidade_max_historica'] = df.groupby(['pdv', 'produto'])['quantidade_total'].transform(
            lambda x: x.expanding().max()
        )
        
        df['quantidade_min_historica'] = df.groupby(['pdv', 'produto'])['quantidade_total'].transform(
            lambda x: x.expanding().min()
        )
        
        return df

    def train_lightgbm_model(self, df):
        print("Treinando modelo LightGBM...")
        
        # Prepare features
        feature_cols = [col for col in df.columns if col not in [
            'pdv', 'produto', 'quantidade_total', 'synthetic_date', 'year_week',
            'categoria', 'tipos', 'label', 'subcategoria', 'premise', 'categoria_pdv'
        ]]
        
        print(f"Features disponíveis: {len(feature_cols)}")
        print(f"Dataset inicial: {df.shape}")
        
        # Remove rows with missing target but be less aggressive with lag features
        df_clean = df.dropna(subset=['quantidade_total'])
        print(f"Após remover target nulo: {df_clean.shape}")
        
        # Only require lag_1 for minimum history
        if 'quantidade_lag_1' in df_clean.columns:
            df_clean = df_clean.dropna(subset=['quantidade_lag_1'])
            print(f"Após remover lag_1 nulo: {df_clean.shape}")
        
        if len(df_clean) == 0:
            raise ValueError("No data remaining after cleaning")
        
        X = df_clean[feature_cols].fillna(0)
        y = df_clean['quantidade_total']
        
        print(f"Shape final - X: {X.shape}, y: {y.shape}")
        
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError(f"Invalid data shape: X={X.shape}, y={y.shape}")
        
        # Split data: use temporal split
        df_clean = df_clean.sort_values(['year', 'week'])
        train_size = int(len(df_clean) * 0.85)
        
        X_train, X_val = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_val = y.iloc[:train_size], y.iloc[train_size:]
        
        print(f"Train: {X_train.shape}, Val: {X_val.shape}")
        
        # LightGBM parameters - more conservative
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'force_row_wise': True
        }
        
        # Train model
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=200,
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(20)]
        )
        
        self.feature_columns = feature_cols
        
        # Validation metrics
        y_pred = self.model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        
        print(f"Validation MAE: {mae:.2f}")
        print(f"Validation RMSE: {rmse:.2f}")
        
        return df_clean

    def generate_predictions(self, df_processed):
        print("Gerando previsões para Janeiro 2023...")
        
        # Get unique PDV-Product combinations from training data
        combinations = df_processed[['pdv', 'produto']].drop_duplicates()
        
        # Filter to most active combinations to reduce output size
        product_activity = df_processed.groupby(['pdv', 'produto'])['quantidade_total'].agg(['count', 'sum']).reset_index()
        product_activity = product_activity[(product_activity['count'] >= 4) & (product_activity['sum'] > 0)]
        combinations = combinations.merge(product_activity[['pdv', 'produto']], on=['pdv', 'produto'])
        
        print(f"Fazendo previsões para {len(combinations)} combinações PDV-Produto")
        
        predictions = []
        
        # Generate predictions for weeks 1-4 of January 2023
        for week in range(1, 5):
            week_predictions = []
            
            for _, combo in combinations.iterrows():
                pdv = combo['pdv']
                produto = combo['produto']
                
                # Get historical data for this combination
                hist_data = df_processed[
                    (df_processed['pdv'] == pdv) & 
                    (df_processed['produto'] == produto)
                ].copy()
                
                if len(hist_data) < 4:  # Need minimum history
                    continue
                
                hist_data = hist_data.sort_values(['year', 'week']).tail(20)  # Last 20 weeks
                
                # Create prediction row
                pred_row = hist_data.iloc[-1].copy()
                pred_row['year'] = 2023
                pred_row['week'] = week
                pred_row['month'] = 1
                pred_row['quarter'] = 1
                pred_row['season'] = 0  # Winter
                
                # Update lag features
                if len(hist_data) >= 1:
                    pred_row['quantidade_lag_1'] = hist_data.iloc[-1]['quantidade_total']
                if len(hist_data) >= 2:
                    pred_row['quantidade_lag_2'] = hist_data.iloc[-2]['quantidade_total']
                if len(hist_data) >= 3:
                    pred_row['quantidade_lag_3'] = hist_data.iloc[-3]['quantidade_total']
                if len(hist_data) >= 4:
                    pred_row['quantidade_lag_4'] = hist_data.iloc[-4]['quantidade_total']
                
                # Update rolling features
                recent_quantities = hist_data['quantidade_total'].tail(8)
                pred_row['quantidade_rolling_mean_2'] = recent_quantities.tail(2).mean()
                pred_row['quantidade_rolling_mean_4'] = recent_quantities.tail(4).mean()
                pred_row['quantidade_rolling_mean_8'] = recent_quantities.mean()
                pred_row['quantidade_rolling_std_2'] = recent_quantities.tail(2).std()
                pred_row['quantidade_rolling_std_4'] = recent_quantities.tail(4).std()
                
                # Prepare features for prediction
                X_pred = pred_row[self.feature_columns].fillna(0).values.reshape(1, -1)
                
                # Make prediction
                pred_quantity = self.model.predict(X_pred)[0]
                pred_quantity = max(0, int(round(pred_quantity)))  # Ensure non-negative integer
                
                week_predictions.append({
                    'semana': week,
                    'pdv': int(pdv),
                    'produto': int(produto),
                    'quantidade': pred_quantity
                })
            
            predictions.extend(week_predictions)
            print(f"Semana {week}: {len(week_predictions)} previsões")
        
        return pd.DataFrame(predictions)

    def run_complete_forecast_pipeline(self):
        # Load and merge data
        df_pdv, df_transactions, df_products = self.import_and_merge_data()
        df_combined = self.data_merge(df_pdv, df_transactions, df_products)
        
        print("Processando dados completos (sem chunks para agregação)...")
        
        # Process all data at once for weekly aggregation to avoid duplicates
        df_weekly = self.weekly_aggregation(df_combined)
        
        # Free memory
        del df_combined
        
        # Check if we have enough data
        if len(df_weekly) == 0:
            raise ValueError("No data remaining after weekly aggregation")
        
        print(f"Dados agregados: {df_weekly.shape}")
        
        # Process feature engineering in chunks if needed
        if len(df_weekly) > self.chunksize * 2:
            print("Processando features em chunks...")
            all_chunks = []
            
            total_chunks = len(df_weekly) // self.chunksize + 1
            for i, start in enumerate(range(0, len(df_weekly), self.chunksize)):
                end = start + self.chunksize
                chunk = df_weekly.iloc[start:end].copy()
                
                # Process chunk
                features_chunk = self.feature_engineering(chunk)
                time_features_chunk = self.time_series_features(features_chunk)
                
                all_chunks.append(time_features_chunk)
                print(f"Processado chunk {i+1}/{total_chunks}")
            
            # Combine all chunks
            df_final = pd.concat(all_chunks, ignore_index=True)
        else:
            # Process all at once if dataset is manageable
            print("Processando features completas...")
            df_features = self.feature_engineering(df_weekly)
            df_final = self.time_series_features(df_features)
        
        # Remove duplicates that might have been created
        df_final = df_final.drop_duplicates(subset=['pdv', 'produto', 'year', 'week'])
        
        print(f"Dataset final processado: {df_final.shape}")
        
        # Check data quality before training
        if len(df_final) == 0:
            raise ValueError("No data remaining after feature processing")
        
        # Train model
        df_processed = self.train_lightgbm_model(df_final)
        
        # Generate predictions
        predictions_df = self.generate_predictions(df_processed)
        
        # Save to CSV
        output_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "previsoes_janeiro_2023.csv")
        predictions_df.to_csv(output_file, index=False)
        print(f"Previsões salvas em: {output_file}")
        print(f"Total de previsões: {len(predictions_df)}")
        
        return df_processed, predictions_df

if __name__ == "__main__":
    forecast_model = TrainForecast()
    results, predictions = forecast_model.run_complete_forecast_pipeline()
    print("Pipeline completo!")
    print(f"Primeiras previsões:\n{predictions.head(10)}")
