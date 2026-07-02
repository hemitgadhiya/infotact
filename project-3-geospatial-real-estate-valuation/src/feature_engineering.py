import numpy as np  # pyrefly: ignore
import pandas as pd  # pyrefly: ignore
import geopandas as gpd  # pyrefly: ignore
from typing import Union

def haversine_distance(
    lat1: Union[float, pd.Series, np.ndarray],
    lon1: Union[float, pd.Series, np.ndarray],
    lat2: Union[float, pd.Series, np.ndarray],
    lon2: Union[float, pd.Series, np.ndarray]
) -> Union[float, pd.Series, np.ndarray]:
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    Returns distance in kilometers.
    """
    # Radius of earth in kilometers
    r = 6371.0
    
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return r * c

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate standard and geospatial features from input dataframe.
    This includes:
    1. Parsing date if it's not already datetime.
    2. Calculating house age and years since last renovation.
    3. Constructing bedroom/bathroom ratios and total rooms.
    4. Calculating square footage ratios (e.g., relative to lot size, neighbors, and per-room sizes).
    5. Constructing distance to Seattle city center and Bellevue center.
    6. Extracting seasonality features (year, month, quarter).
    """
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # 1. Parse date if string
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
        
    sale_year = pd.DatetimeIndex(df['date']).year
    
    # 2. Age & Renovation Features
    # House age at the time of sale
    df['house_age'] = np.maximum(0, sale_year - df['yr_built'])
    
    # Binary renovation flag
    df['is_renovated'] = (df['yr_renovated'] > 0).astype(int)
    
    # Years since renovation (if renovated, else years since built)
    df['years_since_renovation'] = np.where(
        df['yr_renovated'] > 0,
        np.maximum(0, sale_year - df['yr_renovated']),
        df['house_age']
    )
    
    # 3. Room Counts and Ratios
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']
    
    # Avoid division by zero by using np.where
    df['bedrooms_per_bathroom'] = np.where(
        df['bathrooms'] > 0,
        df['bedrooms'] / df['bathrooms'],
        df['bedrooms']  # fallback if bathrooms is 0
    )
    
    # 4. Square Footage Ratios & Indicators
    df['has_basement'] = (df['sqft_basement'] > 0).astype(int)
    
    df['sqft_living_to_lot_ratio'] = df['sqft_living'] / np.maximum(1, df['sqft_lot'])
    
    df['sqft_living_per_bedroom'] = np.where(
        df['bedrooms'] > 0,
        df['sqft_living'] / df['bedrooms'],
        df['sqft_living']
    )
    
    df['sqft_living_per_bathroom'] = np.where(
        df['bathrooms'] > 0,
        df['sqft_living'] / df['bathrooms'],
        df['sqft_living']
    )
    
    # Compare with neighbors
    df['sqft_living_vs_neighbor_ratio'] = df['sqft_living'] / np.maximum(1.0, df['sqft_living15'])
    df['sqft_lot_vs_neighbor_ratio'] = df['sqft_lot'] / np.maximum(1.0, df['sqft_lot15'])
    
    # 5. Distance to Key Hubs (Geospatial)
    # Seattle City Center coordinates: Lat: 47.6062, Lon: -122.3321
    df['dist_to_seattle_center_km'] = haversine_distance(
        df['lat'], df['long'], 47.6062, -122.3321
    )
    
    # Bellevue Center coordinates: Lat: 47.6101, Lon: -122.2015
    df['dist_to_bellevue_center_km'] = haversine_distance(
        df['lat'], df['long'], 47.6101, -122.2015
    )
    
    # 6. Seasonality Features
    df['sale_year'] = sale_year
    df['sale_month'] = pd.DatetimeIndex(df['date']).month
    df['sale_quarter'] = pd.DatetimeIndex(df['date']).quarter
    
    return df

def engineer_geospatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate spatial and geospatial features from input dataframe (e.g. distance to amenities, buffer zones).
    """
    return engineer_features(df)
