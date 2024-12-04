import streamlit as st
import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math
from pathlib import Path

st.set_page_config(layout="wide")


custom_css = """
<style>

    /* Style for color picker */
        div[data-testid="stColorPicker"] label p {
        font-size: 18px !important;  /* Label font size */
    }

    div[data-testid="stColorPicker"] div[role="button"] {
        width: 40px !important;  /* Size of the color picker button */
        height: 40px !important; /* Size of the color picker button */
    }

    div[data-testid="stColorPicker"] input {
        font-size: 18px !important; /* Font size for the input field */
    }

    div[class*="stNumberInput"] label p {
        font-size: 18px !important;
    }

    div[class*="stNumberInput"] input {
        font-size: 16px !important;
    }

    .dataframe {
        width: 100%;  /* Set the width of the table */
        max-height: 600px;  /* Set the maximum height for the table */
        overflow-y: auto;    /* Enable vertical scrolling */
    }
    
    .dataframe th, .dataframe td {
        text-align: center;  /* Center align text */
        padding: 10px;  /* Add padding */
        border: 1px solid #ddd;  /* Add border */
        word-wrap: break-word;  /* Allow text to wrap */
        max-width: 200px;  /* Set a max width for wrapping */
    }

</style>
"""


@st.cache_data
def load_and_arrange_data(scenario):
  
    # Define file paths based on the selected scenario
    if scenario == "SSP245":
        mean_data_path = Path(__file__).parent/"data/mean_ssp245_climate_params_for_new_indian_districts.gpkg"
        max_data_path = Path(__file__).parent/"data/max_ssp245_climate_params_for_new_indian_districts.gpkg"
        min_data_path = Path(__file__).parent/"data/min_ssp245_climate_params_for_new_indian_districts.gpkg"
    elif scenario == "SSP585":
        mean_data_path = Path(__file__).parent/"data/mean_ssp585_climate_params_for_new_indian_districts.gpkg"
        max_data_path = Path(__file__).parent/"data/max_ssp585_climate_params_for_new_indian_districts.gpkg"
        min_data_path = Path(__file__).parent/"data/min_ssp585_climate_params_for_new_indian_districts.gpkg"


    mean_data = gpd.read_file(mean_data_path)
    max_data = gpd.read_file(max_data_path)
    min_data = gpd.read_file(min_data_path)
    

    # Identify common columns
    common_columns = ['District', 'STATE', 'geometry', 'REMARKS', 'State_LGD', 'DISTRICT_L', 'Shape_Leng', 'Shape_Area']

    # Merge data
    merged_data = mean_data.merge(max_data, on=['District', 'STATE', 'geometry', 'REMARKS', 'State_LGD', 'DISTRICT_L', 'Shape_Leng', 'Shape_Area'], suffixes=('_mean', '_max'))

    # Merge with min_data and rename columns
    merged_data = merged_data.merge(min_data, on=common_columns)
    for col in min_data.columns:
        if col not in common_columns:
            merged_data = merged_data.rename(columns={col: f"{col}_min"})

    # Get the list of unique parameter names (without suffixes)
    param_names = [col.rsplit('_', 1)[0] for col in merged_data.columns if col.endswith('_mean')]

    # Create a new column order
    new_order = ['District', 'STATE', 'geometry', 'REMARKS', 'State_LGD', 'DISTRICT_L', 'Shape_Leng', 'Shape_Area']
    for param in param_names:
        new_order.extend([f"{param}_mean", f"{param}_max", f"{param}_min"])

    # Reorder the columns
    merged_data = merged_data[new_order]
    
    return merged_data


def rename_columns_with_prefix(df, og_new_names_dict):

    # Create a new dictionary to store the final mapping of column names
    final_column_names = {}
    
    # Iterate over the columns of the dataframe
    for col in df.columns:
        # Split the column name to check the suffix (_mean, _max, _min)
        if col.endswith('_mean'):
            base_name = col.rsplit('_', 1)[0]  # Remove the '_mean' part
            new_col_name = "Average of " + og_new_names_dict.get(base_name, base_name)
        elif col.endswith('_max'):
            base_name = col.rsplit('_', 1)[0]  # Remove the '_max' part
            new_col_name = "Maximum of " + og_new_names_dict.get(base_name, base_name)
        elif col.endswith('_min'):
            base_name = col.rsplit('_', 1)[0]  # Remove the '_min' part
            new_col_name = "Minimum of " + og_new_names_dict.get(base_name, base_name)
        else:
            new_col_name = og_new_names_dict.get(col, col)  # If no suffix, just use the dictionary or original name
        
        # Add the mapping to the final dictionary
        final_column_names[col] = new_col_name

    # Rename the dataframe columns using the final dictionary
    df = df.rename(columns=final_column_names)
    return df


def simplify_and_prepare_geojson(gdf, tolerance=0.01):
    simplified = gdf.geometry.simplify(tolerance)
    geojson = json.loads(simplified.to_json())
    return geojson

def aggregate_to_state_level(data):
    # Get all columns except geometry and other non-numeric columns
    value_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    value_columns = [col for col in value_columns if col not in ['geometry']]
    
    # Aggregate to state level
    state_data = data.groupby('State')[value_columns].mean().reset_index()
    return state_data

# Create dropdowns for state and district selection
@st.cache_data
def get_state_districts():
    states = sorted(renamed_data[state_col].unique())
    districts = {state: sorted(renamed_data[renamed_data[state_col] == state][district_col].unique()) for state in states}
    return states, districts


# Function to handle image conversion
def save_figure_as_image(figure):
    return figure.to_image(format="png", scale=4)  # Higher scale for better resolution


# Loading the new column names
colums_csv_fn = Path(__file__).parent/"data/renamed_columns.csv"
renamed_columns_data = pd.read_csv(colums_csv_fn)

og_new_names_dict = {}
new_og_names_dict = {}

for og_name, new_name in zip(renamed_columns_data['og_names'], renamed_columns_data['new_names']):
    og_new_names_dict[og_name] = new_name

for new_name, og_name in zip(renamed_columns_data['new_names'], renamed_columns_data['og_names']):
    new_og_names_dict[new_name] = og_name


st.title("Climate Projections for Indian Districts (2021-2040)")
st.markdown("<h4>Explore the Climate Parameters across Indian Districts and States</h4>", unsafe_allow_html=True)

# Sidebar for user input
st.sidebar.header("Select IPCC Pathways and Climate Parameters")


# Sidebar for selecting scenario
scenario = st.sidebar.selectbox(
    "Select IPCC Scenario",
    options=["SSP245", "SSP585"]
)


# Load data
data = load_and_arrange_data(scenario)

# Rename specific columns
data.rename(columns={
  'DISTRICT_L': 'District Code',
  'STATE': 'State'
}, inplace=True)

# Reload data with renamed columns
renamed_data = rename_columns_with_prefix(data, og_new_names_dict)

# Prepare geojson (do this once, but not cached)
geojson = simplify_and_prepare_geojson(data)

# Aggregate to state level
state_data = aggregate_to_state_level(renamed_data)


# Get unique old parameter names
old_param_names = [col.rsplit('_', 1)[0] for col in data.columns if col.endswith('_mean')]

# Get unique parameter names
param_names = [' '.join(col.split(' ')[2:]) for col in renamed_data.columns if col.startswith('Average of ')]

parameter = st.sidebar.selectbox(
  "Choose a climate parameter",
  options=param_names
)

og_parameter = new_og_names_dict[parameter] 


stat_type = st.sidebar.radio(
"Select statistic type",
options=['mean', 'max', 'min']
)


stat_type_dict = {'mean':'Average of', 'max': 'Maximum of', 'min': 'Minimum of'}

# Create choropleth map
og_column_name = f"{og_parameter}_{stat_type}"
column_name = f"{stat_type_dict[stat_type]} {parameter}"


# Determine the correct column names for district and state
district_col = 'District'
state_col = 'State'
district_l_col = 'District Code'


# Split the parameter name into multiple lines for the title
prefix = stat_type_dict[stat_type]
split_param = (prefix+' '+parameter).split(' ')
if len(split_param)>5:
    if len(split_param)>10:
        first_line = ' '.join(split_param[:5])
        second_line = ' '.join(split_param[5:10])
        third_line = ' '.join(split_param[10:]) 
        parameter_title = first_line+'<br>'+second_line+'<br>'+third_line
    else:
        first_line = ' '.join(split_param[:5])
        second_line = ' '.join(split_param[5:10])
        parameter_title = first_line+'<br>'+second_line
else:
   parameter_title = prefix+' '+parameter


# Split the parameter name into units for the legend
split_param_for_legend = og_parameter.split('(')
if len(split_param_for_legend)>1:
    legend_title = split_param_for_legend[1].split(')')[0]
    if legend_title[0]=='i':
            legend_title = legend_title[3:]


# Define the custom color scales
custom_negative_scale = ['#543005','#74552B','#957A51','#B59E77','#D6C39D','#F6E8C3',"#F5F5F5"]
custom_positive_scale = ["#F5F5F5", '#c7eae5','#80cdc1','#35978f','#01665e','#003c30']
custom_temp_scale = ["#f5f5f5", "#fddbc7", "#f4a482", "#d65f4d", "#b2182a", "#67001f"]
custom_rh_scale = ["#f5f5f5","#d1e5f0", "#92c5de", "#4392c3", "#2166ac", "#053061"]

custom_precip_params = ['rf', 'cdd', 'r10', 'r20', 'rx1', 'rx5', '5day_events', 'sdii', 'rainy_days']
custom_temp_params = ['tmax', 'tmin', 'wet_bulb', 'mam_csu', 'summer_days', 'hwdi', 'wsdi', 'warm_spells']
custom_rh_params = ['rh']
custom_cdd_params = ['cdd']

# Choose the color scale based on the parameter and data range
if any(param in og_column_name for param in custom_precip_params):

  min_val = data[og_column_name].min()
  max_val = data[og_column_name].max()
  
  if min_val >= 0:
      if any(param in og_column_name for param in custom_cdd_params):
        color_scale = custom_negative_scale[::-1]
      else:
        color_scale = custom_positive_scale
  elif max_val <= 0:
      if any(param in og_column_name for param in custom_cdd_params):
        color_scale = custom_positive_scale[::-1]
      else:
        color_scale = custom_negative_scale
  else:
      if any(param in og_column_name for param in custom_cdd_params):
        # Create a diverging color scale
        neg_range = abs(min_val)
        pos_range = max_val
        total_range = neg_range + pos_range
        neg_colors = custom_positive_scale[::-1][:-1]  # Exclude the middle color
        pos_colors = custom_negative_scale[::-1][1:]   # Exclude the middle color
        
        color_scale = [
            (0, neg_colors[0]),
            (neg_range / total_range / 2, neg_colors[-1]),
            (neg_range / total_range, "#F5F5F5"),
            (neg_range / total_range + pos_range / total_range / 2, pos_colors[0]),
            (1, pos_colors[-1])
        ]
      else:
        # Create a diverging color scale
        neg_range = abs(min_val)
        pos_range = max_val
        total_range = neg_range + pos_range
        neg_colors = custom_negative_scale[:-1]  # Exclude the middle color
        pos_colors = custom_positive_scale[1:]   # Exclude the middle color
        
        color_scale = [
            (0, neg_colors[0]),
            (neg_range / total_range / 2, neg_colors[-1]),
            (neg_range / total_range, "#F5F5F5"),
            (neg_range / total_range + pos_range / total_range / 2, pos_colors[0]),
            (1, pos_colors[-1])
        ]
elif any(param in og_column_name for param in custom_temp_params):
  color_scale = custom_temp_scale
elif any(param in og_column_name for param in custom_rh_params):
  color_scale = custom_rh_scale
else:
  color_scale = "Viridis"



## Create the map using go.Choroplethmapbox
#fig = go.Figure(go.Choroplethmapbox(
#    geojson=geojson,
#    locations=renamed_data.index,
#    z=renamed_data[column_name],
#    colorscale=color_scale,
#    marker_opacity=1,
#    marker_line_width=0.5,
#    colorbar=dict(
#        title=legend_title,  # Set the colorbar title
#        titlefont=dict(size=20),  # Set the font size for the colorbar title
#        tickfont=dict(size=20)  # Set the font size for the colorbar ticks
#    ),
#    hovertemplate="<b>%{customdata[0]}</b><br>" +
#                  "State: %{customdata[1]}<br>" +
#                  "District (LGD): %{customdata[2]}<br>" +
#                  f"{column_name}: %{{z:.2f}}<extra></extra>",
#    customdata=renamed_data[[district_col, state_col, district_l_col]].values
#))
#
#fig.update_layout(
#    mapbox_style="carto-positron",
#    mapbox_zoom=3.3,
#    mapbox_center={"lat": 22.5937, "lon": 81.9629},
#    width=1000,
#    height=700,
#    margin=dict(b=0),
#    annotations=[
#        dict(
#            text=f"{parameter_title}",
#            x=0.5,  # Center horizontally over the map
#            y=1.02,  # Adjust vertical positioning (1.0 is the top of the map, values above 1 move the text higher)
#            xref="paper",  # x-coordinate relative to the paper (figure)
#            yref="paper",  # y-coordinate relative to the paper (figure)
#            showarrow=False,  # Remove the arrow
#            font=dict(size=20),  # Adjust font size
#            xanchor='center',  # Center the text horizontally
#            yanchor='bottom'   # Align text above the map
#        )
#    ]
#)


# Display the map with custom width and height
#st.plotly_chart(fig, use_container_width=True)

# Create a download button
#st.download_button(
#    label="Download Map as PNG",
#    data=save_figure_as_image(fig),
#    file_name=f"{og_parameter}_map.png",
#    mime="image/png"
#)


#st.text('')
#st.text('')

# Create a histogram of the values across all districts
#fig_hist = px.histogram(renamed_data, x=column_name, title='')

# Update layout with custom font sizes
#fig_hist.update_layout(
#    width=1000,
#    height=500,
#    annotations=[
#        dict(
#            text="Distribution of Values Across All Districts",
#            x=0.45,  # Center horizontally over the map
#            y=1.1,  # Adjust vertical positioning (1.0 is the top of the map, values above 1 move the text higher)
#            xref="paper",  # x-coordinate relative to the paper (figure)
#            yref="paper",  # y-coordinate relative to the paper (figure)
#            showarrow=False,  # Remove the arrow
#            font=dict(size=20),  # Adjust font size
#            xanchor='center',  # Center the text horizontally
#            yanchor='top'   # Align text above the map
#        )
#    ],
#    xaxis={
#        'title': {
#            'text': 'Value',
#            'font': {'size': 20}  # Increase the font size of the x-axis title
#        },
#        'tickfont': {'size': 20}  # Increase the font size of the x-axis ticks
#    },
#    yaxis={
#        'title': {
#            'text': 'Frequency',
#            'font': {'size': 20}  # Increase the font size of the y-axis title
#        },
#        'tickfont': {'size': 20}  # Increase the font size of the y-axis ticks
#    }
#)

# Display the interactive histogram below the first map
#st.plotly_chart(fig_hist, use_container_width=True)
#
#st.text('')
#st.text('')
#st.text('')
#st.text('')



# Create columns for the histogram and colormap interface
#col1, col2 = st.columns([2, 1])
#
## Placeholder for the map (we'll update this later)
#fig_placeholder = col1.empty()
#
## Display the colormap interface in the second column
#with col2:
#    st.markdown('<center><h3>Create your own colormap</h3></center>', unsafe_allow_html=True)
#
#    with st.form("color_form"):
#        st.markdown(custom_css, unsafe_allow_html=True)
#        color1 = st.color_picker('Select bottom color', '#ff0000')
#        color2 = st.color_picker('Select mid color', '#00ff00')
#        color3 = st.color_picker('Select top color', '#0000ff')
#        
#        min_value = st.number_input('Min value', min_value=-1000.0, max_value=1000.0, value=0.3)
#        mid_value = st.number_input('Mid value', min_value=-1000.0, max_value=1000.0, value=0.8)
#        max_value = st.number_input('Max value', min_value=-1000.0, max_value=1000.0, value=1.6)
#
#        form_submitted = st.form_submit_button('Submit')
#
#if form_submitted:
#    colors = [color1, color2, color3]
#    ranges = [min_value, mid_value, max_value]
#    
#    # Create a continuous colormap
#    cmap = mcolors.LinearSegmentedColormap.from_list("", colors)
#    
#    # Convert the cmap to a colorscale
#    colorscale = []
#    for i in np.linspace(0, 1, 100):
#        rgb = cmap(i)[:3]
#        colorscale.append([i, f"rgb({rgb[0]*255}, {rgb[1]*255}, {rgb[2]*255})"])
#    
#    # Create a new colorbar object
#    colorbar = dict(
#        title=legend_title,
#        titlefont=dict(size=20),
#        tickfont=dict(size=20),
#        tick0=ranges[0],
#        dtick=(ranges[2] - ranges[0]) / 2,  # Adjust this if necessary
#        tickvals=[ranges[0], ranges[1], ranges[2]],  # Ensure these values are correct
#        ticktext=[f'{ranges[0]}', f'{ranges[1]}', f'{ranges[2]}'],  # Ensure these match tickvals
#        ticks='outside'
#    )
#    
#    data_temp = renamed_data.copy()
#    data_temp[column_name] = np.clip(data_temp[column_name], ranges[0], ranges[2])
#
#    # Create a new figure with the updated colorscale and colorbar
#    fig_temp = go.Figure(go.Choroplethmapbox(
#        geojson=geojson,
#        locations=data_temp.index,
#        z=data_temp[column_name],
#        colorscale=colorscale,
#        marker_opacity=1,
#        marker_line_width=0.5,
#        colorbar=colorbar,
#        hovertemplate="<b>%{customdata[0]}</b><br>" +
#                    "State: %{customdata[1]}<br>" +
#                    "District (LGD): %{customdata[2]}<br>" +
#                    f"{column_name}: %{{z:.2f}}<extra></extra>",
#        customdata=data_temp[[district_col, state_col, district_l_col]].values
#        ))
#
#    fig_temp.update_layout(
#        mapbox_style="carto-positron",
#        mapbox_zoom=3.3,
#        mapbox_center={"lat": 22.5937, "lon": 81.9629},
#        width=1000,
#        height=700,
#        margin=dict(b=0),
#        annotations=[
#            dict(
#                text=f"{parameter_title}",
#                x=0.5,  # Center horizontally over the map
#                y=1.02,  # Adjust vertical positioning (1.0 is the top of the map, values above 1 move the text higher)
#                xref="paper",  # x-coordinate relative to the paper (figure)
#                yref="paper",  # y-coordinate relative to the paper (figure)
#                showarrow=False,  # Remove the arrow
#                font=dict(size=20),  # Adjust font size
#                xanchor='center',  # Center the text horizontally
#                yanchor='bottom'   # Align text above the map
#            )
#        ]
#    )
#
#    # Display the updated figure
#    fig_placeholder.plotly_chart(fig_temp, use_container_width=True)
#
#    # Create a download button outside the form
#    st.download_button(
#        label="Download Map as PNG",
#        data=save_figure_as_image(fig_temp),
#        file_name=f"{og_parameter}_custom_colormap.png",
#        mime="image/png"
#    )


#st.text('')
#st.text('')
#st.text('')
#st.text('')


# Display data table
#st.subheader("District-level Data (Sorted in Descending Order)")

## Create a copy of the data with only the columns we need
district_display_data = renamed_data[[district_col, state_col, district_l_col, column_name]].copy()
#
## Sort the data in descending order based on the selected column and round to 2 decimal places
district_display_data = district_display_data.sort_values(by=column_name, ascending=False).round(2)
#
## Reset the index to show the rank
district_display_data = district_display_data.reset_index(drop=True)
district_display_data.index += 1  # Start the index from 1 instead of 0
#
## Append legend_title to the last column name
last_column_name = district_display_data.columns[-1]  # Get the last column name
#district_display_data.rename(columns={last_column_name: f"{last_column_name} ({legend_title})"}, inplace=True)
#
## Display the sorted data
#st.markdown(custom_css, unsafe_allow_html=True)
#st.markdown('<div class="dataframe">{}</div>'.format(district_display_data.to_html(escape=False, index=True)), unsafe_allow_html=True)
#st.text('')
#
## Create a download button for the DataFrame
#csv = district_display_data.to_csv(index=True)  # Convert DataFrame to CSV
#st.download_button(
#  label="Download Data as CSV",
#  data=csv,
#  file_name='district_data.csv',
#  mime='text/csv',
#  key='download-csv'
#)
#
##st.text('')
##st.text('')
##st.text('')
#
#
## State-level Analysis
#st.header("State-level Analysis")
#
## Display state-level data
#st.subheader("State-level Data (Sorted in Descending Order)")
#state_display_data = state_data[[state_col, column_name]].sort_values(by=column_name, ascending=False).round(2)
#
## Reset the index to show the rank starting from 1
#state_display_data = state_display_data.reset_index(drop=True)
#state_display_data.index += 1  # Start the index from 1 instead of 0
#
## Append legend_title to the last column name
#last_column_name = state_display_data.columns[-1]  # Get the last column name
#state_display_data.rename(columns={last_column_name: f"{last_column_name} ({legend_title})"}, inplace=True)
#
## Display the sorted data
#st.markdown(custom_css, unsafe_allow_html=True)
#st.markdown('<div class="dataframe">{}</div>'.format(state_display_data.to_html(escape=False, index=True)), unsafe_allow_html=True)
#st.text('')
#
## Create a download button for the DataFrame
#csv = state_display_data.to_csv(index=True)  # Convert DataFrame to CSV
#st.download_button(
#  label="Download Data as CSV",
#  data=csv,
#  file_name='state_data.csv',
#  mime='text/csv',
#  key='download-csv-2'
#)
#
#st.text('')


## Display boxplots for each state
#st.subheader(f"Distribution of Values by State")
#
#st.text('')
#st.text('')
#
## Calculate median values for each state and sort
#state_medians = renamed_data.groupby(state_col)[column_name].median().sort_values(ascending=False)
#state_order = state_medians.index.tolist()
#
#fig_box = px.box(renamed_data, x=state_col, y=column_name,
#                 category_orders={state_col: state_order},
#                 hover_data=[district_col],  # Include district information in hover data
#                 labels={district_col: "District"})  # Label for the district column in hover
#
#fig_box.update_traces(
#    hovertemplate="<b>State:</b> %{x}<br>" +
#                  "<b>District:</b> %{customdata[0]}<br>" +
#                  f"<b>{column_name}:</b> %{{y:.2f}}<extra></extra>"
#)
#
#fig_box.update_layout(
#    xaxis={
#        'title': {
#            'text': 'State',
#            'font': {'size': 20}  # Increase the font size of the x-axis title
#        },
#        'tickfont': {'size': 16}  # Increase the font size of the x-axis ticks
#    },
#    yaxis={
#        'title': {
#            'text': legend_title,
#            'font': {'size': 20}  # Increase the font size of the y-axis title
#        },
#        'tickfont': {'size': 20}  # Increase the font size of the y-axis ticks
#    },
#    annotations=[
#        dict(
#            text=f"{parameter_title}",
#            x=0.5,  # Center horizontally over the map
#            y=0.98,  # Adjust vertical positioning (1.0 is the top of the map, values above 1 move the text higher)
#            xref="paper",  # x-coordinate relative to the paper (figure)
#            yref="paper",  # y-coordinate relative to the paper (figure)
#            showarrow=False,  # Remove the arrow
#            font=dict(size=20),  # Adjust font size
#            xanchor='center',  # Center the text horizontally
#            yanchor='bottom'   # Align text above the map
#        )
#    ],
#    width=1000,  # Increase width
#    height=1000,  # Increase height
#    margin=dict(l=50, r=50, t=80, b=50)  # Increased top margin
#)
#
#st.plotly_chart(fig_box)
#
#
#st.text('')
#st.text('')


st.header("State-District Analysis")

states, districts = get_state_districts()

col1, col2 = st.columns(2)

with col1:
  selected_state = st.selectbox("Select a state", ["All States"] + states)

with col2:
  if selected_state != "All States":
      selected_district = st.selectbox("Select a district", ["All Districts"] + districts[selected_state])
  else:
      selected_district = "All Districts"
      st.text("")
      st.text("")
      st.text("Please select a state first")


# Function to create focused map with calculated zoom and zoom cap
def create_focused_map(data, geojson, column_name, color_scale, selected_state, selected_district):
    if selected_state != "All States":
        data_filtered = data[data[state_col] == selected_state]
        if selected_district != "All Districts":
            data_filtered = data_filtered[data_filtered[district_col] == selected_district]
    else:
        data_filtered = data

    fig = go.Figure(go.Choroplethmapbox(
        geojson=geojson,
        locations=data_filtered.index,
        z=data_filtered[column_name],
        colorscale=color_scale,
        marker_opacity=1,
        marker_line_width=0.5,
        colorbar_title=legend_title,
        colorbar=dict(
            title=legend_title,  # Set the colorbar title
            titlefont=dict(size=20),  # Set the font size for the colorbar title
            tickfont=dict(size=20)  # Set the font size for the colorbar ticks
        ),
        hovertemplate="<b>%{customdata[0]}</b><br>" +
                      "State: %{customdata[1]}<br>" +
                      "District (LGD): %{customdata[2]}<br>" +
                      f"{column_name}: %{{z:.2f}}<extra></extra>",
        customdata=data_filtered[[district_col, state_col, district_l_col]].values
    ))

    flag1,flag2 = '',''

    # Calculate bounds (minx, miny, maxx, maxy) for the selected region
    if selected_state != "All States":
        flag1 = 'state'
        if selected_district != "All Districts":
            flag2 = 'one_district'
            district_boundary = data[(data[state_col] == selected_state) & (data[district_col] == selected_district)]['geometry'].iloc[0]
            bounds = district_boundary.bounds
        else:
            flag2 = 'all_districts'
            state_boundary = data[data[state_col] == selected_state]['geometry'].union_all()
            bounds = state_boundary.bounds
    else:
        bounds = data.total_bounds  # Bounds for all data if no specific state/district selected

    # Calculate center of the bounds
    center_lat = (bounds[1] + bounds[3]) / 2  # (miny + maxy) / 2
    center_lon = (bounds[0] + bounds[2]) / 2  # (minx + maxx) / 2

    # Calculate zoom level based on the bounds size (distance between corners)
    def calculate_zoom(bounds, map_width=1000, map_height=700):
        max_lat_diff = bounds[3] - bounds[1]  # maxy - miny
        max_lon_diff = bounds[2] - bounds[0]  # maxx - minx
        
        # Approximate zoom calculation formula based on latitude/longitude differences and map size
        zoom_lat = math.log2(360 / max_lat_diff) - 1.5
        zoom_lon = math.log2(360 / max_lon_diff) - 1.5

        # Use the more restrictive zoom (so both axes fit into the view)
        return (zoom_lat+zoom_lon)/2.0

    # Get the zoom level
    zoom = calculate_zoom(bounds)

    # Set a minimum zoom level to prevent extreme zooming out for large areas like India
    MIN_ZOOM = 3.2  # Adjust this based on the desired zoom for large areas like India
    zoom = max(zoom, MIN_ZOOM)

    # Set the map center and zoom level
    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=zoom,
        mapbox_center={"lat": center_lat, "lon": center_lon},
        width=1000,
        height=700,
        margin=dict(b=0),
        annotations=[
            dict(
                text=f"{parameter_title} - Focused View",
                x=0.5,  # Center horizontally over the map
                y=1.02,  # Adjust vertical positioning (1.0 is the top of the map, values above 1 move the text higher)
                xref="paper",  # x-coordinate relative to the paper (figure)
                yref="paper",  # y-coordinate relative to the paper (figure)
                showarrow=False,  # Remove the arrow
                font=dict(size=20),  # Adjust font size
                xanchor='center',  # Center the text horizontally
                yanchor='bottom'   # Align text above the map
            )
        ]
    )

    return (fig, flag1, flag2, selected_state, selected_district)


# Create the focused map
focused_fig, flag1, flag2, sel_state, sel_dist = create_focused_map(renamed_data, geojson, column_name, color_scale, selected_state, selected_district)

# Display the focused map
st.plotly_chart(focused_fig, use_container_width=True)

if flag1 == 'state' and flag2 == 'all_districts':
    # Create a download button
    st.download_button(
        label="Download Map as PNG",
        data=save_figure_as_image(focused_fig),
        file_name=f"{og_parameter}_{sel_state}_map.png",
        mime="image/png"
    )
elif flag1 == 'state' and flag2 == 'one_district':
    # Create a download button
    st.download_button(
        label="Download Map as PNG",
        data=save_figure_as_image(focused_fig),
        file_name=f"{og_parameter}_{sel_dist}_map.png",
        mime="image/png"
    )

st.text('')
st.text('')


# Display focused data
if selected_state != "All States":
    st.subheader(f"Data for {selected_state}")
    state_data = renamed_data[renamed_data[state_col] == selected_state].round(2)
    if selected_district != "All Districts":
        st.write(f"Showing data for {selected_district}")
        district_data = state_data[state_data[district_col] == selected_district]
        district_data = district_data[[district_col, state_col, district_l_col, column_name]]
        district_data.rename(columns={last_column_name: f"{last_column_name} ({legend_title})"}, inplace=True)
        st.markdown(custom_css, unsafe_allow_html=True)
        st.markdown('<div class="dataframe">{}</div>'.format(district_data.to_html(escape=False, index=True)), unsafe_allow_html=True)

        st.text('')

        csv = district_data.to_csv(index=True)  # Convert DataFrame to CSV
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name='state_district_data.csv',
            mime='text/csv',
            key='download-csv-3'
        )


    else:
        st.write("Showing data for all districts in the selected state")
        state_data = state_data[[district_col, state_col, district_l_col, column_name]]
        state_data.rename(columns={last_column_name: f"{last_column_name} ({legend_title})"}, inplace=True)
        st.markdown(custom_css, unsafe_allow_html=True)
        st.markdown('<div class="dataframe">{}</div>'.format(state_data.to_html(escape=False, index=True)), unsafe_allow_html=True)

        st.text('')

        csv = state_data.to_csv(index=True)  # Convert DataFrame to CSV
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name='state_district_data.csv',
            mime='text/csv',
            key='download-csv-4'
        )

st.text('')
st.text('')


st.write(f'''
    <a target="_blank" href="https://climateprojections.azimpremjiuniversity.edu.in/">
        <button>
            National level scenario
        </button>
    </a>
    ''',
    unsafe_allow_html=True
)
#
#
##st.header("Extreme Event Analysis")
#
## Get the list of columns starting from "Annual Wet Bulb Temperature" to the last column
#start_col = 'Average of Annual Wet Bulb Temperature'
#start_idx = list(renamed_data.columns).index(start_col)  # Find the index of the start column
#extreme_event_params = renamed_data.columns[start_idx:]  # Slice the columns from the start index to the end
#
## Create a dropdown to choose the extreme event
#selected_event = st.selectbox("Select an extreme event", extreme_event_params)
#
#
## Split the parameter name into multiple lines for the title
#prefix = ' '.join(selected_event.split(' ')[:2])
#
#split_param = (selected_event).split(' ')
#if len(split_param)>5:
#    if len(split_param)>10:
#        first_line = ' '.join(split_param[:5])
#        second_line = ' '.join(split_param[5:10])
#        third_line = ' '.join(split_param[10:]) 
#        parameter_extreme_title = first_line+'<br>'+second_line+'<br>'+third_line
#    else:
#        first_line = ' '.join(split_param[:5])
#        second_line = ' '.join(split_param[5:10])
#        parameter_extreme_title = first_line+'<br>'+second_line
#else:
#   parameter_extreme_title = prefix+' '+selected_event
#
#
## Calculate threshold values for the selected event
#threshold_value = renamed_data[selected_event].quantile(0.95)
#
## Identify districts prone to the selected extreme event
#prone_districts = []
#for index, row in renamed_data.iterrows():
#    if row[selected_event] > threshold_value:
#        prone_districts.append((index, 1))
#
#custom_data = [[renamed_data.loc[district, 'District'], renamed_data.loc[district, 'State'], renamed_data.loc[district, 'District Code']] for district in renamed_data.index]
#
## Create a map to highlight the districts with extreme values
#fig_extreme = go.Figure(go.Choroplethmapbox(
#    geojson=geojson,
#    locations=renamed_data.index,
#    z=[1]*len(renamed_data.index),
#    marker_opacity=1,
#    marker_line_width=0.5,
#    colorscale=[[0, "lightgray"], [1, "red"]],
#    showscale=False
#))
#
## Highlight the prone districts
#prone_districts_z = [1 if district in [d[0] for d in prone_districts] else 0 for district in renamed_data.index]
#fig_extreme.add_trace(go.Choroplethmapbox(
#    geojson=geojson,
#    locations=renamed_data.index,
#    z=prone_districts_z,
#    marker_opacity=1,
#    marker_line_width=0.5,
#    colorscale=[(0, "lightgray"), (1, "red")],
#    showscale=False,
#    hovertemplate="<b>%{customdata[0]}</b><br>" +
#                  "State: %{customdata[1]}<br>" +
#                  "District (LGD): %{customdata[2]}<br>" +
#                  "<extra></extra>",
#    customdata=custom_data
#))
#
## Update the map with the extreme districts
#fig_extreme.update_layout(
#    mapbox_style="carto-positron",
#    mapbox_zoom=3.3,
#    mapbox_center={"lat": 22.5937, "lon": 81.9629},
#    width=1000,
#    height=850,
#    margin=dict(t=150,b=0),
#    annotations=[
#        dict(
#            text=f"Districts with Extreme Values (>95th percentile) of<br>{parameter_extreme_title}",
#            x=0.5,  # Center horizontally over the map
#            y=1.02,  # Adjust vertical positioning (1.0 is the top of the map, values above 1 move the text higher)
#            xref="paper",  # x-coordinate relative to the paper (figure)
#            yref="paper",  # y-coordinate relative to the paper (figure)
#            showarrow=False,  # Remove the arrow
#            font=dict(size=20),  # Adjust font size
#            xanchor='center',  # Center the text horizontally
#            yanchor='bottom'   # Align text above the map
#        )
#    ]
#)
#
## Display the map
##st.plotly_chart(fig_extreme, use_container_width=True)
#
## Create a download button
##st.download_button(
##    label="Download Map as PNG",
##    data=save_figure_as_image(fig_extreme),
##    file_name=f"{og_parameter}_extremes_map.png",
##    mime="image/png"
##)
