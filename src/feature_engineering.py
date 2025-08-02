import pandas as pd
import numpy as np
from src.utils import calc_mode

def generate_customer_features(df: pd.DataFrame, customers_df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    customers_df = customers_df.copy()
    final_df = customers_df.copy()

    df['price'] = df['price'] * 1000

    # Podstawowe agregaty
    customer_agg = df.groupby('customer_id', observed=True).agg(
        first_purchase_date=('t_dat', 'min'),
        last_purchase_date=('t_dat', 'max'),
        num_baskets=('t_dat', 'nunique'),
        total_spent=('price', 'sum'),
        total_items=('article_id', 'count'),
        mean_price=('price', 'mean'), # średnia cena artykułów klienta
        median_price=('price', 'median'), # mediana
        min_price=('price', 'min'), # najniższa cena
        max_price=('price', 'max'), # najwyższa cena
        price_std=('price', 'std'), # odchylenie standardowe
        price_var=('price', 'var'), # wariancja
        price_skew=('price', pd.Series.skew), # skośność
        price_kurt=('price', pd.Series.kurt), # kurtoza
        unique_articles=('article_id', 'nunique'),
        channels_used=('sales_channel_id', 'nunique'),
        unique_product_types=('product_type_name', 'nunique'),
        unique_garment_groups=('garment_group_name', 'nunique'),
        unique_colour_master=('perceived_colour_master_name', 'nunique'),
        unique_dep_name=('department_name', 'nunique'),
        unique_index_group=('index_group_name', 'nunique'),
        unique_index=('index_name', "nunique"),
        unique_graph_appearance=('graphical_appearance_name', 'nunique'),
        unique_prod=('prod_name', 'nunique'),
        unique_color_group=('colour_group_name', 'nunique'),
        unique_color_value=('perceived_colour_value_name', 'nunique')
    ).reset_index()

    # Wartości modalne
    most_common = df.groupby('customer_id', observed=True).agg(
    most_common_articles=('article_id', calc_mode),
    most_common_channel=('sales_channel_id', calc_mode),
    most_common_product_type=('product_type_name', calc_mode),
    most_common_garment_group=('garment_group_name', calc_mode),
    most_common_colour_master=('perceived_colour_master_name', calc_mode),
    most_common_department=('department_name', calc_mode),
    most_common_index_group=('index_group_name', calc_mode),
    most_common_index=('index_name', calc_mode),
    most_common_graph_appearance=('graphical_appearance_name', calc_mode),
    most_common_prod_name=('prod_name', calc_mode),
    most_common_color_group=('colour_group_name', calc_mode),
    most_common_color_value=('perceived_colour_value_name', calc_mode)
    ).reset_index()

    # Długość i czas relacji
    customer_agg['first_purchase_date'] = pd.to_datetime(customer_agg['first_purchase_date'])
    customer_agg['last_purchase_date'] = pd.to_datetime(customer_agg['last_purchase_date'])
    customer_agg['relationship_lenght'] = (
    customer_agg['last_purchase_date'] - customer_agg['first_purchase_date']
    ).dt.days.fillna(0)

    # Zmienna churn
    cutoff_days = 90
    reference_date = pd.to_datetime(df['t_dat'].max())
    customer_agg['days_since_last_purchase'] = (reference_date - customer_agg['last_purchase_date']).dt.days.astype('int16')
    customer_agg['churn'] = (customer_agg['days_since_last_purchase'] > cutoff_days).astype('int8')

    # Aktywność miesięczna
    df['month'] = pd.to_datetime(df['t_dat']).dt.to_period('M')
    active_months = df.groupby('customer_id',observed=True)['month'].nunique().reset_index(name='active_months')

    # Zakupy weekendowe
    df['is_weekend'] = pd.to_datetime(df['t_dat']).dt.dayofweek >= 5
    weekend_counts = df.groupby('customer_id', observed=True)['is_weekend'].agg(
        weekend_purchases='sum', weekend_total_purchases='count'
    ).reset_index()

    # Najczęstszy dzień tygodnia
    df['weekday'] = pd.to_datetime(df['t_dat']).dt.day_name()
    weekday_mode = df.groupby('customer_id', observed=True)['weekday'].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
    ).reset_index(name='dominant_weekday')

    # Zakupy w konkretnych porach roku
    df['month'] = pd.to_datetime(df['t_dat']).dt.month
    df['season'] = df['month'].map({
        12: 'winter', 1: 'winter', 2: 'winter',
        3: 'spring', 4: 'spring', 5: 'spring',
        6: 'summer', 7: 'summer', 8: 'summer',
        9: 'autumn', 10: 'autumn', 11: 'autumn'
    })
    dominant_season = df.groupby('customer_id', observed=True)['season'].agg(
    lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
    ).reset_index(name='dominant_season')

    # Sklejanie wszystkiego
    dfs_to_merge = [customer_agg, active_months, weekend_counts, 
                    weekday_mode, dominant_season, most_common]

    for other_df in dfs_to_merge:
        final_df = final_df.merge(other_df, on='customer_id', how='left')

    # Oczyszczenie danych i zmiana typów
    final_df = final_df.drop(columns=['first_purchase_date', 'last_purchase_date', 'days_since_last_purchase'], errors='ignore')
    final_df = final_df.astype({
    'customer_id': 'string',
    'FN': 'category',
    'Active': 'category',
    'club_member_status': 'category',
    'fashion_news_frequency': 'category',
    'age': 'float16',
    'num_baskets': 'int16',
    'total_items': 'int16',
    'unique_articles': 'int16',
    'channels_used': 'int8',
    'unique_product_types': 'int16',
    'unique_garment_groups': 'int8',
    'unique_colour_master': 'int8',
    'unique_dep_name': 'int16',
    'unique_index_group': 'int8',
    'unique_index': 'int8',
    'unique_graph_appearance': 'int8',
    'unique_prod': 'int16',
    'unique_color_group': 'int8',
    'unique_color_value': 'int8',
    'relationship_lenght': 'int16',
    'active_months': 'int8',
    'weekend_purchases': 'int16',
    'weekend_total_purchases': 'int16',
    'total_spent': 'float32',
    'mean_price': 'float32',
    'median_price': 'float32',
    'min_price': 'float32',
    'max_price': 'float32',
    'price_std': 'float32',
    'price_var': 'float32',
    'price_skew': 'float32',
    'price_kurt': 'float32',
    'dominant_weekday': 'category',
    'dominant_season': 'category',
    'most_common_articles': 'category',
    'most_common_channel': 'category',
    'most_common_product_type': 'category',
    'most_common_garment_group': 'category',
    'most_common_colour_master': 'category',
    'most_common_department': 'category',
    'most_common_index_group': 'category',
    'most_common_index': 'category',
    'most_common_graph_appearance': 'category',
    'most_common_prod_name': 'category',
    'most_common_color_group': 'category',
    'most_common_color_value': 'category',
    'postal_code': 'category',
    'churn': 'category'
})

    return final_df