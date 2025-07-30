import pandas as pd
import numpy as np

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:

    category_columns = [
        'customer_id',
        'article_id',
        'product_type_no',
        'garment_group_no',
        'perceived_colour_master_id',
        'sales_channel_id',
        'season',
        'weekday',
        'dominant_season',
        'dominant_weekday'
    ]

    for col in category_columns:
        if col in df.columns:
            df[col] = df[col].astype('category')

    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype('category')

    for col in df.columns:
        if df[col].dtype.kind in ('f','i','u'):
            c_min, c_max = df[col].min(), df[col].max()
            if c_min >= np.finfo(np.float16).min and c_max <= np.finfo(np.float16).max:
                df[col] = df[col].astype(np.float16)
            elif c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)

    return df.info()

def filter_customers_with_min_purchases(df: pd.DataFrame,
                                        customers_df: pd.DataFrame,
                                        min_purchases: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    active_customers = df['customer_id'].value_counts()
    active_customers = active_customers[active_customers >= min_purchases].index

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