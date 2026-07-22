import sys
from pathlib import Path
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point

# Add current directory to path so we can import local modules
sys.path.append(str(Path(__file__).parent.parent))

from structured_research.lib.figures import save_vector_figure

def main():
    # Load German states boundaries
    url = 'https://raw.githubusercontent.com/isellsoap/deutschlandGeoJSON/main/2_bundeslaender/3_mittel.geo.json'
    states = gpd.read_file(url)

    # Define TSO mapping
    tso_mapping = {
        '50Hertz': ['DE-BB', 'DE-BE', 'DE-MV', 'DE-SN', 'DE-ST', 'DE-TH'],
        'Amprion': ['DE-NW', 'DE-RP', 'DE-SL', 'DE-HE'],
        'TenneT': ['DE-SH', 'DE-NI', 'DE-HB', 'DE-HH', 'DE-BY'],
        'TransnetBW': ['DE-BW']
    }
    
    # Invert mapping to add to dataframe
    state_to_tso = {}
    for tso, state_ids in tso_mapping.items():
        for state_id in state_ids:
            state_to_tso[state_id] = tso

    states['TSO'] = states['id'].map(state_to_tso)

    # Merge states into TSO zones
    tso_zones = states.dissolve(by='TSO').reset_index()

    # Create figure
    fig, ax = plt.subplots(figsize=(4, 5)) # ~10cm wide
    
    # Colors for zones (colorblind friendly)
    colors = {
        '50Hertz': '#E69F00',   # Orange
        'Amprion': '#56B4E9',   # Light Blue
        'TenneT': '#009E73',    # Green
        'TransnetBW': '#CC79A7' # Pink/Magenta
    }
    
    # Plot zones
    for _, row in tso_zones.iterrows():
        tso = row['TSO']
        # Plot geometry
        gpd.GeoSeries([row['geometry']]).plot(
            ax=ax, 
            color=colors[tso], 
            edgecolor='white', 
            linewidth=0.5,
            alpha=0.8,
            label=tso
        )

    # Plot country border
    germany = states.dissolve()
    germany.plot(ax=ax, facecolor='none', edgecolor='#333333', linewidth=1)

    # Add reference cities
    cities = {
        'Berlin': (13.4050, 52.5200),
        'Munich': (11.5820, 48.1351),
        'Hamburg': (9.9937, 53.5511),
        'Stuttgart': (9.1829, 48.7758)
    }
    
    for city, (lon, lat) in cities.items():
        ax.plot(lon, lat, 'o', color='black', markersize=3)
        ax.text(lon + 0.15, lat + 0.15, city, fontsize=8, color='black', 
                ha='left', va='bottom', weight='bold')

    # Add Campus Jülich
    juelich_lon, juelich_lat = 6.3647, 50.9222
    ax.plot(juelich_lon, juelich_lat, '*', color='red', markersize=8, markeredgecolor='black', markeredgewidth=0.5)
    ax.text(juelich_lon - 0.2, juelich_lat + 0.1, 'Campus Jülich', fontsize=8, color='red', 
            ha='right', va='bottom', weight='bold')

    # Legend
    handles = [plt.Rectangle((0,0),1,1, color=colors[tso], alpha=0.8) for tso in colors.keys()]
    ax.legend(handles, colors.keys(), loc='lower right', frameon=True, fontsize=8, title='TSO Zones', title_fontsize=9)

    ax.set_axis_off()
    fig.tight_layout()

    # Save
    save_vector_figure(fig, 'tso_zone_map', also_png=True)
    print("Successfully generated tso_zone_map")

if __name__ == '__main__':
    main()
