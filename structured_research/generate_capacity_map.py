import sys
from pathlib import Path
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.path.append(str(Path(__file__).parent.parent))
from structured_research.lib.figures import save_vector_figure

def main():
    print("Loading German districts GeoJSON...")
    url = 'https://raw.githubusercontent.com/isellsoap/deutschlandGeoJSON/main/4_kreise/3_mittel.geo.json'
    kreise = gpd.read_file(url)

    print("Loading MaStR wind & solar data...")
    mastr_wind = pd.read_parquet('structured_research/data/mastr_wind.parquet')
    mastr_solar = pd.read_parquet('structured_research/data/mastr_solar.parquet')

    print("Performing spatial join...")
    gdf_wind = gpd.GeoDataFrame(
        mastr_wind, geometry=gpd.points_from_xy(mastr_wind.lon, mastr_wind.lat), crs='EPSG:4326'
    )
    gdf_solar = gpd.GeoDataFrame(
        mastr_solar, geometry=gpd.points_from_xy(mastr_solar.lon, mastr_solar.lat), crs='EPSG:4326'
    )

    wind_joined = gpd.sjoin(gdf_wind, kreise, how='inner', predicate='within')
    solar_joined = gpd.sjoin(gdf_solar, kreise, how='inner', predicate='within')

    kreise['wind_mw'] = kreise.index.map(wind_joined.groupby('index_right')['maxPower'].sum() / 1e3).fillna(0)
    kreise['solar_mw'] = kreise.index.map(solar_joined.groupby('index_right')['maxPower'].sum() / 1e3).fillna(0)

    # Plot 2-panel figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5.5))

    # Jülich coords
    juelich_lon, juelich_lat = 6.3647, 50.9222

    # Left: Onshore Wind
    kreise.plot(
        column='wind_mw',
        ax=ax1,
        cmap='Blues',
        edgecolor='0.7',
        linewidth=0.2,
        legend=True,
        legend_kwds={'label': 'Onshore Wind Capacity (MW)', 'orientation': 'horizontal', 'shrink': 0.75, 'pad': 0.02}
    )
    ax1.plot(juelich_lon, juelich_lat, '*', color='red', markersize=9, markeredgecolor='black', markeredgewidth=0.5, zorder=5)
    ax1.text(juelich_lon - 0.25, juelich_lat + 0.15, 'Campus Jülich', fontsize=8, color='crimson', ha='right', weight='bold')
    ax1.set_title('German Onshore Wind Fleet (MaStR 2023)', fontsize=11, weight='bold')
    ax1.set_axis_off()

    # Right: Solar PV
    kreise.plot(
        column='solar_mw',
        ax=ax2,
        cmap='YlOrRd',
        edgecolor='0.7',
        linewidth=0.2,
        legend=True,
        legend_kwds={'label': 'Solar PV Capacity (MW)', 'orientation': 'horizontal', 'shrink': 0.75, 'pad': 0.02}
    )
    ax2.plot(juelich_lon, juelich_lat, '*', color='red', markersize=9, markeredgecolor='black', markeredgewidth=0.5, zorder=5)
    ax2.text(juelich_lon - 0.25, juelich_lat + 0.15, 'Campus Jülich', fontsize=8, color='crimson', ha='right', weight='bold')
    ax2.set_title('German Solar PV Fleet (MaStR 2023)', fontsize=11, weight='bold')
    ax2.set_axis_off()

    fig.tight_layout()
    save_vector_figure(fig, 'capacity_density_map', also_png=True)
    print("Successfully generated capacity_density_map")

if __name__ == '__main__':
    main()
