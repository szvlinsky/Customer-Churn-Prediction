import pandas as pd
import numpy as np
from src.utils import describe_columns

def generate_customer_features(df: pd.DataFrame, customers_df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    customers_df = customers_df.copy()

    # Podstawowe agregaty
    customer_agg = df.groupby('customer_id').agg(
        first_purchase_date=('t_dat', 'min'),
        last_purchase_date=('t_dat', 'max'),
        num_baskets=('t_dat', 'nunique'),
        total_spent=('price', 'sum'),
        total_items=('price', 'count'),
        unique_articles=('article_id', 'nunique'),
        unique_product_types=('product_type_no', 'nunique'),
        unique_garment_groups=('garment_group_no', 'nunique'),
        unique_colors=('perceived_colour_master_id', 'nunique'),
        channels_used=('sales_channel_id', 'nunique')
    ).reset_index()

    # Czas relacji i długość
    customer_agg['first_purchase_date'] = pd.to_datetime(customer_agg['first_purchase_date'])
    customer_agg['last_purchase_date'] = pd.to_datetime(customer_agg['last_purchase_date'])
    customer_agg['relationship_lenght'] = (
    customer_agg['last_purchase_date'] - customer_agg['first_purchase_date']
).dt.days.fillna(0)

    # Informacje o koszykach zakupowych
    basket_stats = df.groupby(['customer_id', 't_dat']).agg(
        basket_value=('price', 'sum'),
        basket_channel=('sales_channel_id', 'first')
    ).reset_index()

    basket_value_stats = basket_stats.groupby('customer_id')['basket_value'].agg(
        max_basket_value='max',
        avg_basket_value='mean'
    ).reset_index()

    basket_counts = basket_stats.groupby(['customer_id', 'basket_channel']).size().unstack(fill_value=0).reset_index()
    basket_counts.columns.name = None
    basket_counts = basket_counts.rename(columns={1: 'offline_baskets', 2: 'online_baskets'})

    # Aktywność miesięczna
    df['month'] = pd.to_datetime(df['t_dat']).dt.to_period('M')
    active_months = df.groupby('customer_id')['month'].nunique().reset_index(name='active_months')

    # Zakupy weekendowe
    df['is_weekend'] = pd.to_datetime(df['t_dat']).dt.dayofweek >= 5
    weekend_counts = df.groupby('customer_id')['is_weekend'].agg(
        weekend_purchases='sum', weekend_total_purchases='count'
    ).reset_index()

    # Najczęstszy dzień tygodnia
    df['weekday'] = pd.to_datetime(df['t_dat']).dt.day_name()
    weekday_mode = df.groupby('customer_id')['weekday'].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
    ).reset_index(name='dominant_weekday')

    # Sklejanie wszystkiego
    enriched_customers_df = customers_df.merge(customer_agg, on='customer_id', how='left')
    enriched_customers_df = enriched_customers_df.merge(basket_value_stats, on='customer_id', how='left')
    enriched_customers_df = enriched_customers_df.merge(basket_counts, on='customer_id', how='left')
    enriched_customers_df = enriched_customers_df.merge(avg_gap, on='customer_id', how='left')
    enriched_customers_df = enriched_customers_df.merge(active_months, on='customer_id', how='left')
    enriched_customers_df = enriched_customers_df.merge(top_color, on='customer_id', how='left')
    enriched_customers_df = enriched_customers_df.merge(weekend_counts, on='customer_id', how='left')
    enriched_customers_df = enriched_customers_df.merge(weekday_mode, on='customer_id', how='left')

    return describe_columns(enriched_customers_df)
