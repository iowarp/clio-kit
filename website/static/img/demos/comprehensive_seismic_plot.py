#!/usr/bin/env python3
"""
Comprehensive GNSS Data Visualization for EarthScope ODSA Station
Creates a multi-panel plot with station metadata and displacement time series
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime
import json

# Load the CSV data
print("Loading seismic data...")
data = pd.read_csv('odsa_data.csv')

# Convert Unix timestamps to datetime
data['datetime'] = pd.to_datetime(data['time'], unit='ms')

# Load metadata from GeoJSON
print("Loading metadata...")
with open('odsa_metadata.geojson', 'r') as f:
    metadata = json.load(f)

station_info = metadata['features'][0]['properties']
coordinates = metadata['features'][0]['geometry']['coordinates']

# Create the comprehensive plot
fig = plt.figure(figsize=(16, 12))
fig.suptitle(f'EarthScope GNSS Station {station_info["station_code"]} - 3D Displacement Time Series\n' +
             f'Network: {station_info["network_code"]}, Location: {station_info["location_code"]}, ' +
             f'Channel: {station_info["channel_code"]}\n' +
             f'Coordinates: {station_info["latitude"]}°N, {abs(float(station_info["longitude"]))}°W',
             fontsize=14, fontweight='bold')

# Sample data for better performance if dataset is large
sample_rate = max(1, len(data) // 10000)  # Sample to ~10k points for plotting
sampled_data = data.iloc[::sample_rate].copy()

# Define colors for each axis
colors = {'east': '#1f77b4', 'north': '#ff7f0e', 'up': '#2ca02c'}

# Create subplots
gs = fig.add_gridspec(4, 2, height_ratios=[3, 3, 3, 1], hspace=0.3, wspace=0.3)

# East displacement subplot
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(sampled_data['datetime'], sampled_data['east'], color=colors['east'], linewidth=0.8, alpha=0.8)
ax1.fill_between(sampled_data['datetime'], 
                 sampled_data['east'] - sampled_data['sigEE'], 
                 sampled_data['east'] + sampled_data['sigEE'], 
                 color=colors['east'], alpha=0.2, label='±1σ uncertainty')
ax1.set_ylabel('East Displacement (m)', fontweight='bold')
ax1.set_title('East-West Movement', fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# North displacement subplot
ax2 = fig.add_subplot(gs[1, :])
ax2.plot(sampled_data['datetime'], sampled_data['north'], color=colors['north'], linewidth=0.8, alpha=0.8)
ax2.fill_between(sampled_data['datetime'], 
                 sampled_data['north'] - sampled_data['sigNN'], 
                 sampled_data['north'] + sampled_data['sigNN'], 
                 color=colors['north'], alpha=0.2, label='±1σ uncertainty')
ax2.set_ylabel('North Displacement (m)', fontweight='bold')
ax2.set_title('North-South Movement', fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Up displacement subplot
ax3 = fig.add_subplot(gs[2, :])
ax3.plot(sampled_data['datetime'], sampled_data['up'], color=colors['up'], linewidth=0.8, alpha=0.8)
ax3.fill_between(sampled_data['datetime'], 
                 sampled_data['up'] - sampled_data['sigUU'], 
                 sampled_data['up'] + sampled_data['sigUU'], 
                 color=colors['up'], alpha=0.2, label='±1σ uncertainty')
ax3.set_ylabel('Up Displacement (m)', fontweight='bold')
ax3.set_title('Vertical Movement', fontweight='bold')
ax3.set_xlabel('Time', fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend()

# Format x-axis for all time series plots
for ax in [ax1, ax2, ax3]:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(sampled_data) // 8000)))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

# Statistics and metadata table
ax4 = fig.add_subplot(gs[3, :])
ax4.axis('off')

# Calculate statistics
stats_text = f"""
DATA STATISTICS & METADATA:
• Total Data Points: {len(data):,} • Sample Rate: 1 Hz • Duration: {(data['datetime'].max() - data['datetime'].min()).days} days
• East: μ={data['east'].mean():.3f}m, σ={data['east'].std():.3f}m, Range=[{data['east'].min():.3f}, {data['east'].max():.3f}]m
• North: μ={data['north'].mean():.3f}m, σ={data['north'].std():.3f}m, Range=[{data['north'].min():.3f}, {data['north'].max():.3f}]m  
• Up: μ={data['up'].mean():.3f}m, σ={data['up'].std():.3f}m, Range=[{data['up'].min():.3f}, {data['up'].max():.3f}]m
• Data Quality: {(data['qChannel'].mean()/1000000):.1f}M avg quality, {len(data) - data.isnull().sum().sum()} complete records
• Station Info: Lat {station_info["latitude"]}°, Lon {station_info["longitude"]}°, Network {station_info["network_code"]}.{station_info["station_code"]}.{station_info["location_code"]}.{station_info["channel_code"]}
• Time Range: {data['datetime'].min().strftime('%Y-%m-%d %H:%M')} to {data['datetime'].max().strftime('%Y-%m-%d %H:%M')} UTC
"""

ax4.text(0.05, 0.8, stats_text, transform=ax4.transAxes, fontsize=10, 
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))

# Add data source info
source_text = "Data Source: EarthScope Consortium | National Data Platform\nGNSS High-Rate (1Hz) Position Time Series"
ax4.text(0.95, 0.2, source_text, transform=ax4.transAxes, fontsize=9,
         horizontalalignment='right', verticalalignment='bottom',
         style='italic', alpha=0.7)

plt.tight_layout()
plt.savefig('comprehensive_gnss_analysis.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('comprehensive_gnss_analysis.pdf', bbox_inches='tight', 
            facecolor='white', edgecolor='none')

print("Comprehensive visualization saved as:")
print("- comprehensive_gnss_analysis.png (high-resolution)")
print("- comprehensive_gnss_analysis.pdf (vector format)")

# Create a summary of key findings
print("\n" + "="*60)
print("GNSS DATA ANALYSIS SUMMARY")
print("="*60)
print(f"Station: {station_info['station_code']} ({station_info['network_code']}.{station_info['location_code']}.{station_info['channel_code']})")
print(f"Location: {station_info['latitude']}°N, {abs(float(station_info['longitude']))}°W")
print(f"Data Points: {len(data):,}")
print(f"Time Span: {(data['datetime'].max() - data['datetime'].min()).days} days")
print(f"Sample Rate: 1 Hz")
print("\nDisplacement Statistics:")
print(f"East:  {data['east'].mean():+.3f} ± {data['east'].std():.3f} m (range: {data['east'].min():.3f} to {data['east'].max():.3f} m)")
print(f"North: {data['north'].mean():+.3f} ± {data['north'].std():.3f} m (range: {data['north'].min():.3f} to {data['north'].max():.3f} m)")
print(f"Up:    {data['up'].mean():+.3f} ± {data['up'].std():.3f} m (range: {data['up'].min():.3f} to {data['up'].max():.3f} m)")
print(f"\nAverage Uncertainties:")
print(f"East:  ±{data['sigEE'].mean():.3f} m")
print(f"North: ±{data['sigNN'].mean():.3f} m") 
print(f"Up:    ±{data['sigUU'].mean():.3f} m")
print("="*60)