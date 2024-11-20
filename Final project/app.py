# Import necessary libraries for data processing, visualization, and creating the app interface
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
# Set up the Streamlit page configuration with a title, icon, layout, and sidebar state
st.set_page_config(
    page_title="Nuclear Explosions Data Explorer",  # Title displayed in the browser tab
    page_icon="💥",  # Emoji icon in the browser tab
    layout="wide",  # Set layout to wide for better use of space
    initial_sidebar_state="expanded",  # Sidebar is expanded by default
)

# Function to load and clean the dataset
@st.cache_data  # This decorator caches the data to improve app performance
def load_data():
    try:
        # Define a dictionary for column mappings and default values
        column_map = {
            'weapon_source_column': 'WEAPON SOURCE COUNTRY',
            'weapon_deployment_column': 'WEAPON DEPLOYMENT LOCATION',
            'year_column': 'Date.Year',
            'month_column': 'Date.Month',
            'day_column': 'Date.Day',
            'yield_upper_column': 'Data.Yeild.Upper',
            'yield_lower_column': 'Data.Yeild.Lower'
        }

        # Try to read the data from the CSV file
        data = pd.read_csv("data.csv", encoding='ISO-8859-1')  # Ensure proper encoding for non-ASCII characters
        
        # List comprehension to collect columns that contain missing values for important columns
        important_columns = [column_map['weapon_source_column'], column_map['weapon_deployment_column']]
        data.dropna(subset=important_columns, inplace=True)  # Drop rows with missing values in important columns

        # Convert the year, month, and day columns to integers and fill missing months and days with default values
        data[column_map['year_column']] = data[column_map['year_column']].astype(int)
        data[column_map['month_column']] = data[column_map['month_column']].fillna(1).astype(int)
        data[column_map['day_column']] = data[column_map['day_column']].fillna(1).astype(int)
        
        # Use list comprehension to calculate the yield range for each row
        data['Yield_Range'] = [
            row[column_map['yield_upper_column']] - row[column_map['yield_lower_column']]
            for _, row in data.iterrows()
        ]
        
        return data, important_columns  # Return the cleaned dataset
    
    except FileNotFoundError:
        print("Error: The file 'data.csv' was not found.")
        return None  # Return None if the file is not found
    except KeyError as e:
        print(f"Error: Missing expected column: {e}")
        return None  # Return None if there's a missing column
    except ValueError as e:
        print(f"Error: Value conversion issue - {e}")
        return None  # Return None if there's a problem with value conversion
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None  # Return None for any other unexpected error


# Load the cleaned data into the variable 'data'
data, _ = load_data()

# Streamlit App Title and Description (Displays at the top of the page)
st.title("💥 Nuclear Explosions Data Explorer (1945-1998)")
st.markdown("""
Explore historical data on nuclear tests conducted around the world. 
Visualizations include maps, bar charts, line graphs, scatter plots, 
and histograms to illustrate the distribution and magnitude of nuclear testing over time.
""")

# Sidebar Filters for the user to select country and year range
st.sidebar.title("Filter Options")
selected_country = st.sidebar.selectbox("Select Country", ["All"] + sorted(data['WEAPON SOURCE COUNTRY'].unique().tolist()))
year_range = st.sidebar.slider("Select Year Range", int(data['Date.Year'].min()), int(data['Date.Year'].max()), (1945, 1998))

# Function to filter data based on selected country and year range
def filter_data(df, country=None, start_year=1945, end_year=1998):
    filtered = df[(df['Date.Year'] >= start_year) & (df['Date.Year'] <= end_year)]  # Filter by year range
    if country and country != "All":  # If a specific country is selected, filter by country
        filtered = filtered[filtered['WEAPON SOURCE COUNTRY'] == country]
    return filtered  # Return the filtered data

# Apply the filter to the dataset based on user selections
filtered_data = filter_data(data, selected_country, year_range[0], year_range[1])

# If a specific country is selected, display data for that country
if selected_country != "All":
    st.subheader(f"Data for {selected_country}")

    # Top 10 Test Purposes for the selected country
    st.header(f"📝 Top 10 Test Purposes for {selected_country}")
    top_purposes = filtered_data[filtered_data['WEAPON SOURCE COUNTRY'] == selected_country]['Data.Purpose'].value_counts().head(10)
    
    # Bar plot for Top 10 Test Purposes
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_purposes.values, y=top_purposes.index, palette="viridis")
    plt.title(f"Top 10 Test Purposes for {selected_country}")
    plt.xlabel("Number of Tests")
    plt.ylabel("Test Purpose")
    st.pyplot(plt)  # Display the plot in the Streamlit app

    # Top 10 Test Locations for the selected country
    st.header(f"📍 Top 10 Test Locations for {selected_country}")
    top_locations = filtered_data[filtered_data['WEAPON SOURCE COUNTRY'] == selected_country]['WEAPON DEPLOYMENT LOCATION'].value_counts().head(10)
    
    # Bar plot for Top 10 Test Locations
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_locations.values, y=top_locations.index, palette="coolwarm")
    plt.title(f"Top 10 Test Locations for {selected_country}")
    plt.xlabel("Number of Tests")
    plt.ylabel("Test Location")
    st.pyplot(plt)  # Display the plot

    # Top 10 Data Sources for the selected country
    st.header(f"🔗 Top 10 Data Sources for {selected_country}")
    top_sources = filtered_data[filtered_data['WEAPON SOURCE COUNTRY'] == selected_country]['Data.Source'].value_counts().head(10)
    
    # Bar plot for Top 10 Data Sources
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_sources.values, y=top_sources.index, palette="Blues")
    plt.title(f"Top 10 Data Sources for {selected_country}")
    plt.xlabel("Number of Tests")
    plt.ylabel("Data Source")
    st.pyplot(plt)  # Display the plot

else:
    # If "All" countries are selected, show global top 10 for purposes, locations, and data sources
    
    # Top 10 Test Purposes globally
    st.header("📝 Top 10 Test Purposes")
    top_purposes = filtered_data['Data.Purpose'].value_counts().head(10)
    
    # Bar plot for global Top 10 Test Purposes
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_purposes.values, y=top_purposes.index, palette="viridis")
    plt.title("Top 10 Test Purposes")
    plt.xlabel("Number of Tests")
    plt.ylabel("Test Purpose")
    st.pyplot(plt)

    # Top 10 Test Locations globally
    st.header("📍 Top 10 Test Locations")
    top_locations = filtered_data['WEAPON DEPLOYMENT LOCATION'].value_counts().head(10)
    
    # Bar plot for global Top 10 Test Locations
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_locations.values, y=top_locations.index, palette="coolwarm")
    plt.title("Top 10 Test Locations")
    plt.xlabel("Number of Tests")
    plt.ylabel("Test Location")
    st.pyplot(plt)

    # Top 10 Data Sources globally
    st.header("🔗 Top 10 Data Sources")
    top_sources = filtered_data['Data.Source'].value_counts().head(10)
    
    # Bar plot for global Top 10 Data Sources
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_sources.values, y=top_sources.index, palette="Blues")
    plt.title("Top 10 Data Sources")
    plt.xlabel("Number of Tests")
    plt.ylabel("Data Source")
    st.pyplot(plt)

# Map Visualization of Nuclear Test Sites
st.header("🌍 Map of Nuclear Test Sites")


# Create the Plotly Express scatter map
fig = px.scatter_mapbox(
    filtered_data,
    lat="Location.Cordinates.Latitude",
    lon="Location.Cordinates.Longitude",
    hover_name="WEAPON DEPLOYMENT LOCATION",  # Use the deployment location for hover
    hover_data={
        "Data.Name": True,  # Test Name
        "Data.Yeild.Lower": True,  # Lower Yield
        "Data.Yeild.Upper": True,  # Upper Yield
        "Date.Year": True,  # Test Year
        "WEAPON SOURCE COUNTRY": True,  # Weapon Source Country
    },
    color_discrete_sequence=["red"],  # Color of the markers
    zoom=1,
    height=600,
    title="Nuclear Test Sites (1945-1998)"
)

# Update the map layout
fig.update_layout(
    mapbox_style="carto-darkmatter",  # Set the map style
    mapbox_zoom=1,
    mapbox_center={"lat": 20, "lon": 10},  # Initial map center
    margin={"r": 0, "t": 30, "l": 0, "b": 0}
)

# Display the Plotly map in Streamlit
st.plotly_chart(fig)


# Group by location and count the number of tests in each location (heatmap density)
location_counts = filtered_data.groupby(['Location.Cordinates.Latitude', 'Location.Cordinates.Longitude']).size().reset_index(name='Test Count')

# Plotly Express Heatmap
st.header("🌐 Heatmap of Test Density by Location")

fig = px.density_mapbox(
    location_counts,
    lat="Location.Cordinates.Latitude",
    lon="Location.Cordinates.Longitude",
    z="Test Count",  # The density value (number of tests at each location)
    color_continuous_scale="Viridis",  # Color scale for intensity
    title="Heatmap of Nuclear Test Density",
    hover_data=["Test Count"],  # Hover to show the number of tests at each location
    height=600,
    zoom=1,  # Initial zoom level
)

fig.update_layout(
    mapbox_style="carto-darkmatter",  # Set the map style
    mapbox_zoom=1,  # Zoom level for the entire map
    mapbox_center={"lat": 20, "lon": 10},  # Initial map center
)

# Display the Plotly heatmap in Streamlit
st.plotly_chart(fig)

# Line Chart showing the number of nuclear tests over time
st.header("📊 Number of Nuclear Tests Over Time")
tests_per_year = filtered_data.groupby('Date.Year').size().reset_index(name='Count')  # Group data by year and count tests

# Plot the number of tests per year
plt.figure(figsize=(10, 5))
sns.lineplot(data=tests_per_year, x='Date.Year', y='Count', marker="o", color="red")
plt.title("Number of Nuclear Tests Over Time")
plt.xlabel("Year")
plt.ylabel("Number of Tests")
st.pyplot(plt)

# Bar Chart showing the number of tests by country
st.header("🏴 Tests by Country")
tests_by_country = filtered_data['WEAPON SOURCE COUNTRY'].value_counts().head(10)  # Count tests per country

# Plot the number of tests by country
plt.figure(figsize=(10, 5))
sns.barplot(x=tests_by_country.values, y=tests_by_country.index, palette="coolwarm")
plt.title("Top Countries by Number of Tests")
plt.xlabel("Number of Tests")
st.pyplot(plt)

# Scatter Plot showing the relationship between yield and number of tests over time
st.header("🔬 Yield vs. Number of Tests")
plt.figure(figsize=(10, 6))
sns.scatterplot(data=filtered_data, x='Date.Year', y='Yield_Range', hue='WEAPON SOURCE COUNTRY', palette="Set2", size='Yield_Range', sizes=(20, 200))
plt.title("Scatter Plot: Yield vs. Number of Tests")
plt.xlabel("Year")
plt.ylabel("Yield Range (Kilotons)")
st.pyplot(plt)


# Histogram showing the distribution of nuclear yield (lower bound)
st.header("🔍 Yield Distribution")
plt.figure(figsize=(10, 5))
sns.histplot(filtered_data['Data.Yeild.Lower'], kde=True, color="blue", bins=20)
plt.title("Distribution of Nuclear Yield (Lower Bound)")
plt.xlabel("Yield (Kiloton)")
st.pyplot(plt)

# Display summary statistics of the data
st.header("📈 Summary Statistics")
st.markdown("Summary statistics of nuclear tests based on the selected filters.")
summary_df = filtered_data[['WEAPON SOURCE COUNTRY', 'Data.Yeild.Lower', 'Data.Yeild.Upper']].describe()
st.dataframe(summary_df)  # Display the summary statistics in a table

# Error Handling for empty data: Show a warning if no data is available for the selected filters
if filtered_data.empty:
    st.warning("No data available for the selected filters. Please try adjusting the filters.")

# Footer with additional information
st.sidebar.markdown("---")
st.sidebar.write("Data Source: Nuclear Explosions Dataset (1945-1998)")
