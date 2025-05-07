# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 09:42:44 2025

@author: natha
"""

import pandas as pd
import nfl_data_py as nfl
import matplotlib.pyplot as plt
import seaborn as sns

# Load play-by-play data for seasons 2021-2023
seasons = [2021, 2022, 2023]
pbp = nfl.import_pbp_data(seasons)

# Filter for pass plays with route and coverage data
pbp_pass = pbp[
    (pbp['play_type'] == 'pass') &
    pbp['route'].notna() &
    pbp['defense_coverage_type'].notna() &
    pbp['defense_man_zone_type'].notna()
]

# Define 20-yard field segments based on yardline_100
def field_segment(yardline_100):
    if yardline_100 <= 20:
        return 'Own Red Zone'
    elif yardline_100 <= 40:
        return 'Own Gold Zone'
    elif yardline_100 <= 60:
        return 'Midfield'
    elif yardline_100 <= 80:
        return 'Opp Gold Zone'
    else:
        return 'Opp Red Zone'

pbp_pass['field_segment'] = pbp_pass['yardline_100'].apply(field_segment)
columns = ['yardline_100', 'down', 'goal_to_go','ydstogo', 'yards_gained', 'shotgun',
           'pass_length','pass_location','air_yards','score_differential','epa','complete_pass',
           'cp','cpoe','temp','wind','pass','xpass','pass_oe','defenders_in_box',
           'number_of_pass_rushers','defense_personnel','ngs_air_yards','time_to_throw',
           'was_pressure','route','defense_man_zone_type','defense_coverage_type','field_segment']
passfiltered = pbp_pass[columns]
print(passfiltered.info())
#--------------------------------------------------------------------------------
# Dummy variable extraction

passfiltered.loc[:, 'zone'] = passfiltered['defense_man_zone_type'].apply(lambda x: 1 if x.lower() == 'zone_coverage' else 0)

def extract_num_dbs(personnel):
    match = re.search(r'(\d+)\s*DB', personnel)
    if match:
        return int(match.group(1))
    else:
        return 0

passfiltered['num_dbs'] = passfiltered['defense_personnel'].apply(extract_num_dbs)

print(passfiltered[['defense_personnel', 'num_dbs']].head(10))
"everything through here is good"
#--------------------------------------------------------------------------------
#------------------------------- Visuals ----------------------------------------
#--------------------------------------------------------------------------------
# comp prob and epa by route visuals
import matplotlib.pyplot as plt

route_cp = passfiltered.groupby('route')['cp'].mean().sort_values()

plt.figure(figsize=(12, 8))
plt.barh(route_cp.index, route_cp.values, color='skyblue')
plt.xlabel('Average Completion Probability (cp)')
plt.ylabel('Route Type')
plt.title('Average Completion Probability by Route Type (2021-2023)')
plt.show()
#-----------
import matplotlib.pyplot as plt
import numpy as np

route_stats = passfiltered.groupby('route').agg({'cp': 'mean', 'epa': lambda x: x[passfiltered['complete_pass'] == 1].mean()})
route_stats = route_stats.sort_values('cp')

fig, ax = plt.subplots(figsize=(12, 8))
width = 0.4
y = np.arange(len(route_stats))

ax.barh(y - width/2, route_stats['cp'], height=width, color='skyblue', label='Average Completion Probability (cp)')
ax.barh(y + width/2, route_stats['epa'], height=width, color='orange', label='Mean EPA on Completion')

ax.set_yticks(y)
ax.set_yticklabels(route_stats.index)
ax.set_xlabel('Value')
ax.set_title('Completion Probability vs. EPA on Completion by Route Type (2021-2023)')
ax.legend(loc='upper right')

plt.show()
#------------
import matplotlib.pyplot as plt
import numpy as np

route_stats = passfiltered.groupby('route').agg({
    'cp': 'mean', 
    'epa': lambda x: x[passfiltered['complete_pass'] == 1].mean()
})
route_stats['weighted_points_added'] = route_stats['cp'] * route_stats['epa']
route_stats = route_stats.sort_values('weighted_points_added')

fig, ax = plt.subplots(figsize=(12, 8))
ax.barh(route_stats.index, route_stats['weighted_points_added'], color='purple')
ax.set_xlabel('Weighted Points Added (cp * epa)')
ax.set_ylabel('Route Type')
ax.set_title('Weighted Points Added by Route Type (2021-2023)')

plt.show()
#--------------------------------------
# comp % by air yards
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import linregress

air_yards_cp = passfiltered.groupby('ngs_air_yards')['cp'].mean().reset_index().dropna()

slope, intercept, r_value, p_value, std_err = linregress(air_yards_cp['ngs_air_yards'], air_yards_cp['cp'])

plt.figure(figsize=(12, 8))
sns.lineplot(x='ngs_air_yards', y='cp', data=air_yards_cp, color='teal', label='Average Completion Probability')
plt.plot(air_yards_cp['ngs_air_yards'], intercept + slope * air_yards_cp['ngs_air_yards'], color='red', linestyle='--', label=f'Trend Line (Slope: {slope:.4f})')

plt.xlabel('NGS Air Yards')
plt.ylabel('Average Completion Probability (cp)')
plt.title('Average Completion Probability by NGS Air Yards (2021-2023)')
plt.legend()
plt.show()
#-------------------------------------
# cp by def in box
import matplotlib.pyplot as plt

defenders_cp = passfiltered.groupby('defenders_in_box')['cp'].mean().sort_index()

plt.figure(figsize=(12, 8))
plt.bar(defenders_cp.index, defenders_cp.values, color='coral')
plt.xlabel('Number of Defenders in the Box')
plt.ylabel('Average Completion Probability (cp)')
plt.title('Average Completion Probability by Number of Defenders in the Box (2021-2023)')
plt.show()
#-------------------------------------
# cp num dbs
import matplotlib.pyplot as plt

dbs_cp = passfiltered.groupby('num_dbs')['cp'].mean().sort_index()

plt.figure(figsize=(12, 8))
plt.bar(dbs_cp.index, dbs_cp.values, color='lightgreen')
plt.xlabel('Number of Defensive Backs (DBs)')
plt.ylabel('Average Completion Probability (cp)')
plt.title('Average Completion Probability by Number of Defensive Backs (2021-2023)')
plt.show()
#-------------------------------------
# cp of routes vs man/zone
import matplotlib.pyplot as plt
import seaborn as sns

route_zone_cp = passfiltered.groupby(['route', 'zone'])['cp'].mean().unstack(fill_value=0)

plt.figure(figsize=(14, 8))
route_zone_cp.plot(kind='barh', stacked=False, color=['orange', 'skyblue'], ax=plt.gca())
plt.xlabel('Average Completion Probability (cp)')
plt.ylabel('Route Type')
plt.title('Completion Probability by Route and Man/Zone Coverage (2021-2023)')
plt.legend(['Man', 'Zone'], loc='upper right')
plt.show()
#--------------------------------------
# cp, epa, wpa against coverage combos
coverage_types = ['2_MAN', 'COVER_0', 'COVER_1', 'COVER_2', 'COVER_3', 'COVER_4', 'COVER_6', 'PREVENT']
coverage_colors = ['skyblue', 'orange', 'green', 'red', 'purple', 'black', 'pink', 'lightgreen']

for coverage in coverage_types:
    plt.figure(figsize=(12, 8))
    
    # Filter data for the current coverage type
    coverage_data = passfiltered[passfiltered['defense_coverage_type'] == coverage]
    
    # Create a list of unique routes
    routes = coverage_data['route'].unique()
    
    # Loop through each route and calculate cp for both man (0) and zone (1)
    for route in routes:
        man_cp = coverage_data[(coverage_data['route'] == route) & (coverage_data['zone'] == 0)]['cp'].mean()
        zone_cp = coverage_data[(coverage_data['route'] == route) & (coverage_data['zone'] == 1)]['cp'].mean()
        
        # Bar chart for each route
        plt.barh(route, man_cp, color='lightcoral', label='Man', edgecolor='black')
        plt.barh(route, zone_cp, color='skyblue', label='Zone', edgecolor='black')
        plt.text(man_cp + 0.01, route, f'{man_cp:.2f}', va='center', ha='left', color='black', fontweight='bold')
        plt.text(zone_cp + 0.01, route, f'{zone_cp:.2f}', va='center', ha='left', color='black', fontweight='bold')
    
    plt.xlabel('Completion Probability (cp)')
    plt.ylabel('Route')
    plt.title(f'Completion Probability for Routes by Coverage Type: {coverage}')
    plt.legend(['Man', 'Zone'])
    plt.show()
#--------------------
coverage_types = ['2_MAN', 'COVER_0', 'COVER_1', 'COVER_2', 'COVER_3', 'COVER_4', 'COVER_6', 'PREVENT']
coverage_colors = ['skyblue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'lightgreen']

# Ensure the 'zone' column is present
passfiltered.loc[:, 'zone'] = passfiltered['defense_man_zone_type'].apply(lambda x: 1 if x.lower() == 'zone' else 0)

# Loop through the coverage types and routes
coverage_types = ['2_MAN', 'COVER_0', 'COVER_1', 'COVER_2', 'COVER_3', 'COVER_4', 'COVER_6', 'PREVENT']
coverage_colors = ['red', 'orange', 'green', 'blue', 'purple', 'brown', 'pink', 'lightgreen']

# Ensure the 'zone' column is present (man = 0, zone = 1)
passfiltered.loc[:, 'zone'] = passfiltered['defense_man_zone_type'].apply(lambda x: 1 if x.lower() == 'zone' else 0)

# Loop through each coverage type and plot bars for each route
"""for coverage in coverage_types:
    plt.figure(figsize=(12, 8))
    
    # Plot for each route
    for idx, route_tuple in enumerate(route_coverages.index):
        route = route_tuple[0]  # Extract the route name from MultiIndex
        # Get cp and epa for this route and coverage
        cp = route_coverages.loc[route_tuple, coverage] if coverage in route_coverages.columns else 0
        epa = route_coverages_epa.loc[route, coverage] if coverage in route_coverages_epa.columns else 0
        
        # Calculate WPA (cp * epa)
        weighted_points = cp * epa

        # Check for man or zone coverage
        zone_value = passfiltered.loc[passfiltered['route'] == route, 'zone'].values[0]  # 1 for zone, 0 for man
        color = 'green' if zone_value == 1 else 'red'  # Zone -> green, Man -> red
        
        # Positioning the bars (each route will have two bars: man and zone)
        plt.barh(route, weighted_points, color=color)
        plt.text(weighted_points + 0.01, route, f'{weighted_points:.2f}', va='center', ha='left', color='black', fontweight='bold')

    plt.xlabel('Weighted Points Added (WPA)')
    plt.ylabel('Route')
    plt.title(f'Route WPA by Coverage Type: {coverage}')
    plt.legend(['Man Coverage (Red)', 'Zone Coverage (Green)'])
    plt.show()
"""
#--------------------------------------------
# Calculate WPA for each route-coverage combination
wpa_df = passfiltered.groupby(['route', 'defense_coverage_type']).apply(
    lambda group: (group['cp'] * group['epa']).mean()).reset_index(name='WPA')

# Sort the WPA DataFrame by WPA and get the top 20 route-coverage combinations
top_20_wpa = wpa_df.sort_values(by='WPA', ascending=False).head(20)

# Plot the top 20 route-coverage combinations
plt.figure(figsize=(12, 8))
bars = plt.barh(top_20_wpa['route'] + ' - ' + top_20_wpa['defense_coverage_type'], top_20_wpa['WPA'], color='skyblue')

# Add the WPA values on the bars
for bar in bars:
    plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, 
             f'{bar.get_width():.2f}', va='center', ha='left', color='black', fontweight='bold')

plt.xlabel('Weighted Points Added (WPA)')
plt.ylabel('Route - Coverage Type')
plt.title('Top 20 Route-Coverage Combinations by WPA')
plt.gca().invert_yaxis()  # To display the highest WPA at the top
plt.show()
#---------------------------------------------
# cp vs xpass
# Group by xpass and calculate the average cp for each xpass value
cp_vs_xpass = passfiltered.groupby('xpass')['cp'].mean().reset_index()

# Plot cp vs xpass on a line chart
plt.figure(figsize=(10, 6))
plt.plot(cp_vs_xpass['xpass'], cp_vs_xpass['cp'], marker='o', color='b', linestyle='-', linewidth=2, markersize=6)

# Add labels and title
plt.xlabel('xpass')
plt.ylabel('Completion Probability (cp)')
plt.title('Completion Probability vs xpass')

# Show the plot
plt.grid(True)
plt.show()
"no pattern found"
#-------------------------------------
# Create a figure with subplots for each route
plt.figure(figsize=(12, 8))

# Iterate over each route
for i, route in enumerate(route_coverages.index):
    plt.subplot(len(route_coverages.index), 1, i + 1)  # Create a subplot for each route
    
    # Extract cp values for each coverage type for this route
    route_cp = passfiltered[passfiltered['route'] == route].groupby('defense_coverage_type')['cp'].mean()
    
    # Only plot if route_cp is non-empty
    if not route_cp.empty:
        # Plot cp vs coverage type
        route_cp.plot(kind='bar', color='skyblue', width=0.7)
        
        # Add labels and title
        plt.xlabel('Coverage Type')
        plt.ylabel('Completion Probability (cp)')
        plt.title(f'Completion Probability of {route} by Coverage Type')
        
        # Add value labels to the bars
        for j, value in enumerate(route_cp):
            plt.text(j, value + 0.01, f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    else:
        # If route_cp is empty, display a message
        plt.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=12, color='red')

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
