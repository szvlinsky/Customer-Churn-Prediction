import pandas as pd
import numpy as np

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    before = df.memory_usage(deep=True).sum()
    
    for col in df.columns:
        if col not in ['price', 'age', 't_dat']:
            df[col] = df[col].astype('category')
    
    after = df.memory_usage(deep=True).sum()
    print(f"[optimize_dtypes] Memory reduced from {before / 1_048_576:.2f} MB to {after / 1_048_576:.2f} MB")

    return df

def filter_customers_with_min_purchase_days(df: pd.DataFrame,
                                            customers_df: pd.DataFrame,
                                            min_days: int = 3
                                            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    purchase_days = df.groupby('customer_id', observed=True)['t_dat'].nunique()
    active_customers = purchase_days[purchase_days >= min_days].index
    df_filtered = df[df['customer_id'].isin(active_customers)].copy()
    customers_filtered = customers_df[customers_df['customer_id'].isin(active_customers)].copy()

    return df_filtered, customers_filtered

def delete_columns(df: pd.DataFrame, customer_df: pd.DataFrame) -> None:
    columns_to_drop = [
        'postal_code',
        'garment_group_no',
        'section_no',
        'index_group_no',
        'index_code',
        'department_no',
        'perceived_colour_master_id',
        'perceived_colour_value_id',
        'colour_group_code',
        'graphical_appearance_no',
        'product_type_no',
        'product_code'
    ]

    df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)
    customer_df.drop(columns=[col for col in columns_to_drop if col in customer_df.columns], inplace=True)