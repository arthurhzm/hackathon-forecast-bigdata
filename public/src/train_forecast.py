import os
import glob
import pandas as pd

class TrainForecast:
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

    

    def run_complete_forecast_pipeline(self):
        df_pdv, df_transactions, df_products = self.import_and_merge_data()
        df_combined = self.data_merge(df_pdv, df_transactions, df_products)

if __name__ == "__main__":

    forecast_model = TrainForecast()
    results, predictions = forecast_model.run_complete_forecast_pipeline()
    print(results, predictions)