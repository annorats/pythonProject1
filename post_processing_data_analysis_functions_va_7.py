import pandas as pd
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as ax
from matplotlib.dates import DateFormatter
from matplotlib.dates import MinuteLocator, DateFormatter
from scipy.interpolate import interp1d
from matplotlib.ticker import ScalarFormatter
import re
import mplcursors
from matplotlib.widgets import Button
from matplotlib.legend_handler import HandlerLine2D
from scipy.optimize import curve_fit


import pandas as pd
import os


def load_csv_files(folder_path):
    """
    Load all CSV files in the specified directory into a single DataFrame.
    """
    df_list = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path)
            df_list.append(df)
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df


def process_df(combined_df):
    combined_df['Time Stamp'] = pd.to_datetime(combined_df['Time Stamp'])
    return combined_df


def split_by_sensor_name(df):
    """
    Split the DataFrame into a dictionary of DataFrames, keyed by channel number.
    """
    sensor_name_dfs = {sensor_name: data for sensor_name, data in df.groupby('Sensor Name')}
    return sensor_name_dfs

def label_channels(df,mapping_dict):
    df['Sensor Name'] = df['Channel'].map(mapping_dict)

def covert_datetime(df):
    df["Time Stamp"] = pd.to_datetime(df["Time Stamp"], format='mixed', dayfirst=True)
    return df

def load_dat_file(filepath):
    """
    Load temperature and resistance data from a given filename.
    """
    return pd.read_csv(filepath, skiprows=2, delim_whitespace=True, names=['Temperature', 'Resistance'])


def extract_cof_file(file_path):
    coefficients = {}
    z_limits = {}
    resistance_limits = {}
    current_range = None

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('FIT RANGE:'):
                current_range = int(line.split(':')[1].strip())
                coefficients[current_range] = []
            elif 'C(' in line:
                coefficients[current_range].append(float(line.split(':')[1].strip()))
            elif 'Zlower' in line:
                z_lower = float(line.split(':')[1].strip())
            elif 'Zupper' in line:
                z_upper = float(line.split(':')[1].strip())
                z_limits[current_range] = (z_lower, z_upper)
            elif 'Lower Resist. limit' in line:
                lower_resist_limit = float(line.split(':')[1].strip())
            elif 'Upper Resist. limit' in line:
                upper_resist_limit = float(line.split(':')[1].strip())
                resistance_limits[current_range] = (lower_resist_limit, upper_resist_limit)

    return coefficients, z_limits, resistance_limits


# Function to calculate temperature using Chebyshev polynomial fit
def chebyshev_temp_fit(resistance, coefficients, z_limits):
    Z = np.log10(resistance)
    ZL, ZU = z_limits
    k = ((Z - ZL) - (ZU - Z)) / (ZU - ZL)
    temperature = sum(Ai * np.cos(i * np.arccos(k)) for i, Ai in enumerate(coefficients))
    return temperature


# Function to determine the temperature range and calculate the temperature
def calculate_temperature(resistance, coefficients, z_limits, resistance_limits):
    for range_id, (lower_limit, upper_limit) in resistance_limits.items():
        if upper_limit >= resistance >= lower_limit:
            return chebyshev_temp_fit(resistance, coefficients[range_id], z_limits[range_id])
    #raise ValueError("Resistance value out of range")
    print(f"Resistance: {resistance}, out of range")
    return None


def plot_cernox_calibrated(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df['Calculated Temp (K)'], df['Resistance'], marker='o', linestyle='', markersize=2, color='dodgerblue', alpha=0.5)
    plt.xlabel('Calculated Temp (K)')
    plt.ylabel('Resistance')
    plt.title('Calculated Temp vs Resistance - calibrated equations')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Calculated Temp vs Resistance - calibrated equations', dpi=300)
    plt.show()


def process_ROx_data(ROx_data_file_path, ROx_coeff_file_path):
    # Load ROx data from file
    df_dat_ROx = load_dat_file(ROx_data_file_path)

    # Extract coefficients and limits from COF file
    coefficients, z_limits, resistance_limits = extract_cof_file(ROx_coeff_file_path)

    # Print header for ROx Data
    print("ROx Data\n")

    # Calculate temperatures and add them to the DataFrame
    df_dat_ROx['Calculated Temp (K)'] = df_dat_ROx['Resistance'].apply(
        lambda r: calculate_temperature(r, coefficients, z_limits, resistance_limits))

    # Print the modified ROx DataFrame
    print("ROx Data\n", df_dat_ROx)

    return df_dat_ROx


def process_Cernox_data(Cernox_data_file_path, Cernox_coeff_file_path):
    # Load Cernox data from file
    df_dat_Cernox = load_dat_file(Cernox_data_file_path)

    # Extract coefficients and limits from COF file
    coefficients, z_limits, resistance_limits = extract_cof_file(Cernox_coeff_file_path)

    # Print header for Cernox Data
    print("Cernox Data\n")

    # Calculate temperatures and add them to the DataFrame
    df_dat_Cernox['Calculated Temp (K)'] = df_dat_Cernox['Resistance'].apply(
        lambda r: calculate_temperature(r, coefficients, z_limits, resistance_limits))

    # Print the modified Cernox DataFrame
    print("Cernox Data\n", df_dat_Cernox)

    # Plot the calibrated Cernox data
    plot_cernox_calibrated(df=df_dat_Cernox)

    return df_dat_Cernox


def process_ROx_fridge_data_chebychev(ROx_coeff_file_path,combined_df):
    coefficients, z_limits, resistance_limits = extract_cof_file(ROx_coeff_file_path)

    # Apply the function only to rows where 'Sensor Name' contains "ROx"
    condition = combined_df['Sensor Name'].str.contains("ROx")
    combined_df.loc[condition, 'Calculated Temp (K)'] = combined_df.loc[condition, 'Resistance'].apply(
        lambda r: calculate_temperature(r, coefficients, z_limits, resistance_limits)
    )

    print("ROx Data\n", combined_df)

    return combined_df


def create_setpoints_list(calibration_data_file_path):
    df_dat = load_dat_file(calibration_data_file_path)
    setpoints_list = df_dat.iloc[1:48:2]['Resistance'].tolist()
    df_setpoints = df_dat.iloc[1:48:2][['Temperature', 'Resistance']]
    return setpoints_list, df_setpoints


def group_stable_regions_by_setpoints(stable_df, setpoints_df):

    # Check if setpoints_df or stable_df is empty
    if setpoints_df.empty or stable_df.empty:
        raise ValueError("setpoints_df or stable_df is empty.")

    # Remove any NaN values from setpoints_df['Resistance']
    setpoints_df = setpoints_df.dropna(subset=['Resistance'])

    # Create an empty column to store the closest setpoint for each stable resistance value
    stable_df['Closest_setpoint'] = np.nan

    # Iterate through each value in stable_df['Resistance']
    for i in range(len(stable_df)):
        stable_value = stable_df['Resistance'].iloc[i]

        # Calculate the absolute difference between the stable value and each setpoint
        differences = abs(setpoints_df['Resistance'] - stable_value)

        # Find the index of the minimum difference, which corresponds to the closest setpoint
        if not differences.empty:
            closest_index = differences.idxmin()
            closest_setpoint = setpoints_df['Resistance'].loc[closest_index]

            # Assign the closest setpoint to the new column
            stable_df.at[i, 'Closest_setpoint'] = closest_setpoint

    # Group the stable regions by the closest setpoint and flatten the result back to a DataFrame
    grouped_stable_regions = stable_df.groupby('Closest_setpoint', as_index=False).apply(lambda x: x)

    # Initialise an empty DataFrame to store the ranges
    ranges_df = pd.DataFrame(columns=['Setpoint', 'Start_Timestamp', 'End_Timestamp'])

    # Iterate through each group to find the start and end times
    for setpoint, group in stable_df.groupby('Closest_setpoint'):
        # Ensure the group is sorted by time (assume 'Timestamp' column exists)
        group = group.sort_values(by='Time Stamp')

        # Find the start and end times for the current setpoint
        start_time = group['Time Stamp'].iloc[0]
        end_time = group['Time Stamp'].iloc[-1]

        # Create a DataFrame with the new row
        new_row = pd.DataFrame({
            'Setpoint': [setpoint],
            'Start_Timestamp': [start_time],
            'End_Timestamp': [end_time]
        })

        #stable_df['Setpoint'] = setpoints_df['Resistance']
        # Concat new row with ranges_df
        ranges_df = pd.concat([ranges_df, new_row], ignore_index=True)
        stable_df['Setpoint'] = stable_df['Closest_setpoint']
    return stable_df, ranges_df

def identify_stable_regions(df, window_size, min_duration, tolerance_fraction, setpoints_df):

    # Convert "Time Stamp" to datetime
    df["Time Stamp"] = pd.to_datetime(df["Time Stamp"], format='mixed', dayfirst=True)

    # Calculate rolling mean and std for the "Resistance" column
    df["Rolling_Mean"] = df["Resistance"].rolling(window=window_size).mean()
    df["Rolling_Std"] = df["Resistance"].rolling(window=window_size).std()

    # Initialise variables to track stable regions
    stable_regions = []
    current_region = []
    prev_mean = None
    prev_std = None

    # Iterate through the DataFrame to identify stable regions
    for i in range(window_size, len(df)):
        current_mean = df["Rolling_Mean"].iloc[i]
        current_std = df["Rolling_Std"].iloc[i]

        if prev_mean is not None and prev_std is not None:
            # Calculate the absolute change in mean and std from the previous window
            mean_change = abs(current_mean - prev_mean)
            std_change = abs(current_std - prev_std)

            # Check if changes are within the stability threshold
            stability_threshold = current_mean * tolerance_fraction
            if mean_change < stability_threshold and std_change < stability_threshold:
                current_region.append(df.iloc[i])
            else:
                if len(current_region) >= min_duration:
                    stable_regions.append(current_region)
                current_region = []

        # Update previous window's mean and std
        prev_mean = current_mean
        prev_std = current_std

    # If there's a region still being tracked at the end of the loop, save it
    if len(current_region) >= min_duration:
        stable_regions.append(current_region)

    # Compile the stable regions into a DataFrame
    stable_df = pd.concat([pd.DataFrame(region) for region in stable_regions], ignore_index=True)
    stable_df, ranges_df = group_stable_regions_by_setpoints(stable_df, setpoints_df)

    return stable_df, ranges_df


def calculate_average_resistance(filtered_data):
    #Calculate the average resistance for the filtered data.
    return filtered_data['Resistance'].mean()


def filter_by_range(uncal_df, start, end, timestamp_column):
    # Filter the uncalibrated dataframe based on a given timestamp range
    uncal_df[timestamp_column] = pd.to_datetime(uncal_df[timestamp_column], format='%d/%m/%Y %H:%M:%S')
    ranges_df = uncal_df[(uncal_df[timestamp_column] >= start) & (uncal_df[timestamp_column] <= end)]
    return ranges_df


def filter_data_by_ranges(df, ranges_df, timestamp_column):
    # Filter the combined dataframe based on the timestamp ranges from the ranges dataframe.
    filtered_data = pd.concat([filter_by_range(df, row['Start_Timestamp'], row['End_Timestamp'], timestamp_column) for _, row in ranges_df.iterrows()])
    return filtered_data

def filter_and_calculate_averages(df, ranges_df, timestamp_column):
    # Filter the uncalibrated dataframe based on the timestamp ranges and calculate average resistance.

    results = []
    for _, row in ranges_df.iterrows():
        filtered_data = filter_by_range(df, row['Start_Timestamp'], row['End_Timestamp'], timestamp_column)

        # Calculate the average resistance and count the number of resistances
        resistance_count = len(filtered_data)
        average_resistance = calculate_average_resistance(filtered_data)

        # Append the results, including the count of resistances used
        results.append({
            'Setpoint': row['Setpoint'],
            'Start_Timestamp': row['Start_Timestamp'],
            'End_Timestamp': row['End_Timestamp'],
            'Average_Resistance': average_resistance,
            'Resistance_Count': resistance_count
        })

    return pd.DataFrame(results), filtered_data

def calculate_average_timestamp(df, ts1, ts2):
    df[ts1] = pd.to_datetime(df[ts1])
    df[ts2] = pd.to_datetime(df[ts2])
    df['average_timestamp'] = df[ts1] + (df[ts2] - df[ts1]) / 2
    return df

def calibrate_data_chebychev(coeff_file_path,df):
    coefficients, z_limits, resistance_limits = extract_cof_file(coeff_file_path)

    df['Calculated Temp (K)'] = df['Average_Resistance'].apply(
        lambda r: calculate_temperature(r, coefficients, z_limits, resistance_limits)
    )
    calibrated_df = df
    print("Calibrated Data\n", calibrated_df)

    return calibrated_df


def calculate_temp_via_chebychev(coeff_file_path, df, resistance_column_title):
    coefficients, z_limits, resistance_limits = extract_cof_file(coeff_file_path)

    df['Newly Calculated Temp (K)'] = df['newly_calibrated_resistance'].apply(
        lambda r: calculate_temperature(r, coefficients, z_limits, resistance_limits)
    )
    print("Calibrated Data\n", df)

    return df

def plot_xy_columns_from_dfs(dfs, x_column, y_column, title="Plot", xlabel="X-axis", ylabel="Y-axis"):
    """
    Plot specific x and y columns from multiple DataFrames on the same plot with different colors.

    Parameters:
    dfs (dict of str: pd.DataFrame): Dictionary of DataFrames to plot, with names as keys.
    x_column (str): The column name to use for the x-axis from each DataFrame.
    y_column (str): The column name to use for the y-axis from each DataFrame.
    title (str): The title of the plot.
    xlabel (str): The label of the x-axis.
    ylabel (str): The label of the y-axis.
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    for name, df in dfs.items():
        if x_column in df.columns and y_column in df.columns:
            ax.plot(df[x_column], df[y_column], label=name, marker='.', markersize=0.25, linestyle='None')
        else:
            print(f"Columns '{x_column}' or '{y_column}' not found in DataFrame '{name}'")

    # Set labels and title with adjusted font sizes
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12,labelpad=20)
    ax.set_title(title, fontsize=14)

    # Adjust grid style and transparency
    ax.grid(True, linestyle='-', alpha=0.5)

    # Add legend with larger markers
    legend = ax.legend(handler_map={plt.Line2D: HandlerLine2D(numpoints=1)})
    for handle in legend.legend_handles:
        handle.set_markersize(10)  # Adjust the size of legend markers


    # Adjust layout for better appearance
    plt.tight_layout()
    fig.subplots_adjust(left=0.15)
    # Save the plot
    plt.savefig(f"plots\\{title}.png", dpi=1200)

    # Show the plot
    plt.show()


def convert_timestamps_to_continuous(df, timestamp_col, unit='seconds'):
    """
    Convert a column of timestamps into a continuous number format.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    timestamp_col (str): The name of the timestamp column in the DataFrame.
    unit (str): The unit for the continuous time format. Default is 'seconds'.
                Other options are 'minutes', 'hours', 'days'.

    Returns:
    pd.DataFrame: The DataFrame with an additional column for continuous time values.
    """
    # Convert timestamp column to datetime
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Calculate the time difference from the start of the first timestamp
    start_time = df[timestamp_col].min()

    if unit == 'seconds':
        df['continuous_time'+'_'+timestamp_col] = (df[timestamp_col] - start_time).dt.total_seconds()
    elif unit == 'minutes':
        df['continuous_time'+'_'+timestamp_col] = (df[timestamp_col] - start_time).dt.total_seconds() / 60
    elif unit == 'hours':
        df['continuous_time'+'_'+timestamp_col] = (df[timestamp_col] - start_time).dt.total_seconds() / 3600
    elif unit == 'days':
        df['continuous_time'+'_'+timestamp_col] = (df[timestamp_col] - start_time).dt.total_seconds() / 86400
    else:
        raise ValueError("Invalid unit. Choose from 'seconds', 'minutes', 'hours', 'days'.")

    return df


def plot_calibrated_ROx_with_stability(df, stable_df, x_column, y_column, title, xlabel, ylabel):
    convert_timestamps_to_continuous(stable_df, 'Start_Timestamp', unit='seconds')
    convert_timestamps_to_continuous(stable_df, 'End_Timestamp', unit='seconds')
    convert_timestamps_to_continuous(stable_df, 'Time Stamp', unit='seconds')

    convert_timestamps_to_continuous(df, x_column, unit='seconds')
    df[x_column] = df['continuous_time'+'_'+x_column]


    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the primary data
    ax.plot(df[x_column], df[y_column], label=title, marker='.', markersize=0.25, linestyle='None')

    # Plot the stability data with error bars
    ax.errorbar(stable_df['continuous_time'+'_'+'Time Stamp'], stable_df['Setpoint'],
                xerr=[stable_df['continuous_time'+'_'+'Start_Timestamp'], stable_df['continuous_time'+'_'+'End_Timestamp']],
                fmt=".", linestyle='None', markersize=0.2, color='red', ecolor='darkseagreen',
                elinewidth=1.2, capsize=0, label='Stability Data')

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12, labelpad=20)
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle='-', alpha=0.5)


    legend = ax.legend(handler_map={plt.Line2D: HandlerLine2D(numpoints=1)})
    for handle in legend.legendHandles:
        handle.set_rasterized(10)

    plt.tight_layout()
    fig.subplots_adjust(left=0.15)

    plt.savefig(f"plots\\{title}.png", dpi=1200)

    plt.show()

def alt_plot_calibrated_ROx_with_stability(df, stable_df, x_column, y_column, title, xlabel, ylabel):

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the primary data
    ax.plot(df[x_column], df[y_column], label='Raw Calibrated Data', marker='.', markersize=0.25, color='dodgerblue', linestyle='None')

    # Plot the setpoints data
    ax.plot(stable_df['Time Stamp'], stable_df['Setpoint'],
            label='Set Points', marker='o', markersize=0.1, linestyle='None', color='red')

    # # Plot time stamp ranges
    # ax.plot(stable_df['Start_Timestamp'], stable_df['End_Timestamp'],
    #         linestyle='-', markersize=0.2, color='purple', label='Stability Timestamp Range')

    # plot the stable data (with over 0.1 std removed)
    ax.plot(stable_df[x_column], stable_df[y_column], label='stable data (with over 0.1 std removed)', marker='.', markersize=0.25, color='purple', linestyle='None')


    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12, labelpad=20)
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle='-', alpha=0.5)

    legend = ax.legend(handler_map={plt.Line2D: HandlerLine2D(numpoints=1)})
    for handle in legend.legend_handles:
        handle.set_rasterized(10)

    plt.tight_layout()

    # Add legend with larger markers
    legend = ax.legend(handler_map={plt.Line2D: HandlerLine2D(numpoints=1)})
    for handle in legend.legend_handles:
        handle.set_markersize(10)  # Adjust the size of legend markers

    fig.subplots_adjust(left=0.15)

    plt.savefig(f"plots\\{title}.png", dpi=1200)

    plt.show()

def plot_calibrated_ROx_with_stability(df, stable_df, x_column, y_column, title, xlabel, ylabel):
    convert_timestamps_to_continuous(stable_df, 'Start_Timestamp', unit='seconds')
    convert_timestamps_to_continuous(stable_df, 'End_Timestamp', unit='seconds')
    convert_timestamps_to_continuous(stable_df, 'Time Stamp', unit='seconds')

    convert_timestamps_to_continuous(df, x_column, unit='seconds')
    df[x_column] = df['continuous_time' + '_' + x_column]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the primary data
    ax.plot(df[x_column], df[y_column], label=title, marker='.', markersize=0.25, linestyle='None')

    # Plot the stability data with error bars
    ax.errorbar(stable_df['continuous_time' + '_' + 'Time Stamp'], stable_df['Setpoint'],
                xerr=[stable_df['continuous_time' + '_' + 'Start_Timestamp'],
                      stable_df['continuous_time' + '_' + 'End_Timestamp']],
                fmt=".", linestyle='None', markersize=0.2, color='red', ecolor='darkseagreen',
                elinewidth=1.2, capsize=0, label='Stability Data')

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12, labelpad=20)
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle='-', alpha=0.5)

    legend = ax.legend(handler_map={plt.Line2D: HandlerLine2D(numpoints=1)})
    for handle in legend.legend_handles:
        handle.set_rasterized(10)

    plt.tight_layout()
    fig.subplots_adjust(left=0.15)

    plt.savefig(f"plots\\{title}.png", dpi=1200)

    plt.show()


def plot_uncalibrated_ROx(df, filtered_uncal, stable_df, avg_uncal_data, x_column, y_column, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the setpoints data
    ax.plot(stable_df['Time Stamp'], stable_df['Setpoint'],
            label='Set Points', marker='o', markersize=0.1, linestyle='None', color='red')
    # plot the stable data (with over 0.1 std removed)
    ax.plot(stable_df[x_column], stable_df[y_column], label='Stable Calibrated U10391 data', marker='.',
            markersize=0.25, color='purple', linestyle='None')

    # plot uncalibrated data within limits
    ax.plot(filtered_uncal[x_column], filtered_uncal[y_column], label='Stable Uncalibrated U10626 data', marker='.',
            markersize=0.25, color='green', linestyle='None')

    # plot avg_resistance_uncal_df
    ax.plot(avg_uncal_data['average_timestamp'], avg_uncal_data['Average_Resistance'],
            label='Average Stable Uncalibrated U10626 data', marker='.', markersize=1, color='orange', linestyle='None')

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12, labelpad=20)
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle='-', alpha=0.5)

    legend = ax.legend(handler_map={plt.Line2D: HandlerLine2D(numpoints=1)})
    for handle in legend.legend_handles:
        handle.set_rasterized(10)

    plt.tight_layout()

    # Add legend with larger markers
    legend = ax.legend(handler_map={plt.Line2D: HandlerLine2D(numpoints=1)})
    for handle in legend.legend_handles:
        handle.set_markersize(10)  # Adjust the size of legend markers

    fig.subplots_adjust(left=0.15)

    plt.savefig(f"plots\\{title}.png", dpi=1200)

    plt.show()


def plot_calibration_curve(avg_cal_data, avg_uncal_data, xlabel, ylabel, title):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the cal data
    ax.plot(avg_cal_data['Calculated Temp (K)'], avg_cal_data['Average_Resistance'],
            label='Average Calibrated Data', marker='.', markersize=2, linestyle='--', linewidth=1, color='dodgerblue')

    # Plot the uncal data
    ax.plot(avg_uncal_data['Calculated Temp (K)'], avg_uncal_data['Average_Resistance'],
            label='Average originally uncalibrated data', marker='.',
            markersize=2, linestyle='--', linewidth=1, color='purple')

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12, labelpad=20)
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle='-', alpha=0.5)

    legend = ax.legend(handler_map={plt.Line2D: HandlerLine2D(numpoints=1)})
    for handle in legend.legend_handles:
        handle.set_rasterized(10)

    plt.tight_layout()

    # Add legend with larger markers
    legend = ax.legend(handler_map={plt.Line2D: HandlerLine2D(numpoints=1)})
    for handle in legend.legend_handles:
        handle.set_markersize(10)  # Adjust the size of legend markers

    fig.subplots_adjust(left=0.15)

    plt.savefig(f"plots\\{title}.png", dpi=1200)

    plt.show()

def alt_plot_calibration_curve(avg_cal_data, avg_uncal_data, xlabel, ylabel, title):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the cal data
    ax.plot(avg_cal_data['average_timestamp'],
            avg_cal_data['Calculated Temp (K)'],
            label='Average Calibrated Data calculated temp',
            marker='.', markersize=2, linestyle='None', linewidth=1, color='dodgerblue')

    # Plot the uncal data
    ax.plot(avg_uncal_data['average_timestamp'], avg_uncal_data['Newly Calculated Temp (K)'],
            label='Average Newly calibrated data calculated temp',
            marker='.', markersize=2, linestyle='None', linewidth=1, color='purple')

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12, labelpad=20)
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle='-', alpha=0.5)

    legend = ax.legend(handler_map={plt.Line2D: HandlerLine2D(numpoints=1)})
    for handle in legend.legend_handles:
        handle.set_rasterized(10)

    plt.tight_layout()

    # Add legend with larger markers
    legend = ax.legend(handler_map={plt.Line2D: HandlerLine2D(numpoints=1)})
    for handle in legend.legend_handles:
        handle.set_markersize(10)  # Adjust the size of legend markers

    fig.subplots_adjust(left=0.15)

    plt.savefig(f"plots\\{title}.png", dpi=1200)

    plt.show()


def alt_alt_plot_calibration_curve(avg_cal_data, avg_uncal_data, xlabel, ylabel, title):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the cal data
    ax.plot(avg_cal_data['Calculated Temp (K)'], avg_cal_data['Average_Resistance'],
            label='Average Calibrated Data', marker='.', markersize=2, linestyle='--', linewidth=1, color='dodgerblue')

    # Plot the uncal data
    ax.plot(avg_uncal_data['Newly Calculated Temp (K)'], avg_uncal_data['newly_calibrated_resistance'],
            label='Average Newly calibrated data', marker='.',
            markersize=2, linestyle='--', linewidth=1, color='purple')

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12, labelpad=20)
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle='-', alpha=0.5)

    legend = ax.legend(handler_map={plt.Line2D: HandlerLine2D(numpoints=1)})
    for handle in legend.legend_handles:
        handle.set_rasterized(10)

    plt.tight_layout()

    # Add legend with larger markers
    legend = ax.legend(handler_map={plt.Line2D: HandlerLine2D(numpoints=1)})
    for handle in legend.legend_handles:
        handle.set_markersize(10)  # Adjust the size of legend markers

    fig.subplots_adjust(left=0.15)

    plt.savefig(f"plots\\{title}.png", dpi=1200)

    plt.show()


def plot_avg_resistance(df_cal, df_uncal,xlabel='average_timestamp',
                        ylabel='Average_Resistance',
                        title='Average Resistance cal, uncal'):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the cal data
    ax.plot(df_cal['average_timestamp'], df_cal['Average_Resistance'],
            label='Calibrated', marker='o', markersize=2, linestyle='--', linewidth=1, color='dodgerblue')

    # Plot the uncal data
    ax.plot(df_uncal['average_timestamp'], df_uncal['Average_Resistance'],
            label='Uncalibrated', marker='o', markersize=2, linestyle='--', linewidth=1, color='purple')

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12, labelpad=20)
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle='-', alpha=0.5)

    legend = ax.legend(handler_map={plt.Line2D: HandlerLine2D(numpoints=1)})
    for handle in legend.legend_handles:
        handle.set_rasterized(10)

    plt.tight_layout()

    # Add legend with larger markers
    legend = ax.legend(handler_map={plt.Line2D: HandlerLine2D(numpoints=1)})
    for handle in legend.legend_handles:
        handle.set_markersize(10)  # Adjust the size of legend markers

    fig.subplots_adjust(left=0.15)

    plt.savefig(f"plots\\{title}.png", dpi=1200)

    plt.show()



def plot_calibrated_vs_uncalibrated_scatter(df_cal, df_uncal, calibrated_label, uncalibrated_label,
                                       x_label='Uncalibrated Sensor Resistance (Ω)',
                                       y_label='Calibrated Sensor Resistance (Ω)',
                                       title='Scatter Plot of Calibrated vs Uncalibrated Sensor Resistance'):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df_uncal[uncalibrated_label], df_cal[calibrated_label], color='dodgerblue', label='Data Points', marker='.',
               s=10)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12, labelpad=20)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, linestyle='-', alpha=0.5)
    fig.subplots_adjust(left=0.15)

    plt.savefig(f"plots\\{title}.png", dpi=1200)

    plt.show()


def polynomial(x, a, b, c):
    return a * x**2 + b * x + c

def fit_and_plot_polynomial(df_cal, df_uncal, calibrated_label, uncalibrated_label,
                            x_label='Uncalibrated Sensor Resistance (Ω)',
                            y_label='Calibrated Sensor Resistance (Ω)',
                            title='Calibration Curve'):

    # Fit the polynomial model to the data
    popt, _ = curve_fit(polynomial, df_uncal[uncalibrated_label], df_cal[calibrated_label])

    # Generate fitted resistance values for plotting
    df_cal['fitted_resistance'] = polynomial(df_uncal[uncalibrated_label], *popt)

    # Plot the original data points and the fitted curve
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df_uncal[uncalibrated_label], df_cal[calibrated_label], color='dodgerblue', label='Data Points', marker='.', s=10)
    ax.plot(df_uncal[uncalibrated_label], df_cal['fitted_resistance'], color='red', label='Fitted Curve', linewidth=1)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12, labelpad=20)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, linestyle='-', alpha=0.5)
    fig.subplots_adjust(left=0.15)
    plt.savefig(f"plots\\{title}.png", dpi=1200)

    plt.show()

    # Print the fitted equation
    print(f"Calibration Curve: R_cal = {popt[0]}*R_uncal^2 + {popt[1]}*R_uncal + {popt[2]}")

    # Return the fitted model parameters
    return popt

def apply_calibration(df_uncal, uncalibrated_label, popt):

    # Apply the fitted model to uncalibrated resistance values to get calibrated values
    df_uncal['newly_calibrated_resistance'] = polynomial(df_uncal[uncalibrated_label], *popt)
    return df_uncal


def plot_calibration_results(df_cal, df_uncal, uncalibrated_label, calibrated_label, calibrated_res_label,
                             x_label='Uncalibrated Sensor Resistance (Ω)',
                             y_label='Calibrated Sensor Resistance (Ω)',
                             title='Comparison of Original and Calibrated Values'):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Original uncalibrated vs calibrated
    ax.plot(df_uncal[uncalibrated_label], df_cal[calibrated_label], color='green', label='Original Data Points',
               marker='x', linewidth=1)

    # Uncalibrated vs calibrated after applying polynomial
    ax.plot(df_uncal[uncalibrated_label], df_uncal[calibrated_res_label], color='dodgerblue',
            marker='+', label='Calibrated Data Points', linewidth=1)

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12, labelpad=20)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, linestyle='-', alpha=0.5)
    fig.subplots_adjust(left=0.15)
    plt.savefig(f"plots\\{title}.png", dpi=1200)

    plt.show()

def alt_plot_calibration_results(df_cal, df_uncal, uncalibrated_label, calibrated_label, calibrated_res_label,
                             x_label='Uncalibrated Sensor Resistance (Ω)',
                             y_label='Calibrated Sensor Resistance (Ω)',
                             title='Comparison of Original and Calibrated Values'):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Original uncalibrated vs calibrated
    ax.plot(df_uncal[uncalibrated_label], df_cal[calibrated_label], color='green', label='Original Data Points for Calibrated vs Originally Uncalibrated',
               marker='x', linewidth=1)

    # Uncalibrated vs calibrated after applying polynomial
    ax.plot(df_uncal[calibrated_res_label], df_cal[calibrated_label], color='dodgerblue',
            marker='+', label='Calibrated vs newly calibrated (previously uncalibrated sensor', linewidth=1)

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12, labelpad=20)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, linestyle='-', alpha=0.5)
    fig.subplots_adjust(left=0.15)
    plt.savefig(f"plots\\{title}.png", dpi=1200)

    plt.show()


def alt_alt_plot_calibration_curve_with_uncertainty(avg_cal_data, avg_uncal_data, xlabel, ylabel, title, pauls_calibration_path):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the calibrated data
    ax.errorbar(avg_cal_data['Calculated Temp (K)'], avg_cal_data['Average_Resistance'],
                xerr=avg_cal_data['total_uncertainty_temp'], yerr=avg_cal_data['total_uncertainty_resistance'], fmt = '.', linestyle='-', capsize=1, label='Average Calibrated Data', color='red')

    # Plot the newly calibrated data with uncertainties
    ax.errorbar(avg_uncal_data['Newly Calculated Temp (K)'], avg_uncal_data['newly_calibrated_resistance'],
                xerr=avg_cal_data['total_uncertainty_temp'], yerr=avg_uncal_data['total_uncertainty_resistance'], fmt = '.', linestyle='-', capsize=1, label='Average Newly Calibrated Data', color='dodgerblue')

    # Plot the comparison data from pauls calibration
    pauls_calibration_df = pd.read_csv(pauls_calibration_path, sep='\t')

    ax.plot(pauls_calibration_df['Kelvin'], pauls_calibration_df['Resistance'],
            label='Pauls calibration curve comparison', color='green', marker='x', markerfacecolor='pink', linestyle='--', markersize='2' )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12, labelpad=20)
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle='-', alpha=0.5)

    legend = ax.legend(handler_map={plt.Line2D: HandlerLine2D(numpoints=1)})
    for handle in legend.legend_handles:
        handle.set_rasterized(10)

    plt.tight_layout()
    fig.subplots_adjust(left=0.15)

    plt.savefig(f"plots\\{title}.png", dpi=1200)
    plt.show()


import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.legend_handler import HandlerLine2D

def alt_alt_log_log_plot_calibration_curve_with_uncertainty(avg_cal_data, avg_uncal_data, xlabel, ylabel, title, pauls_calibration_path):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set x-axis and y-axis to log scale
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Plot the calibrated data
    ax.errorbar(avg_cal_data['Calculated Temp (K)'], avg_cal_data['Average_Resistance'],
                xerr=avg_cal_data['total_uncertainty_temp'], yerr=avg_cal_data['total_uncertainty_resistance'],
                fmt='.', linestyle='-', capsize=1, label='Average Calibrated Data', color='red')

    # Plot the newly calibrated data with uncertainties
    ax.errorbar(avg_uncal_data['Newly Calculated Temp (K)'], avg_uncal_data['newly_calibrated_resistance'],
                xerr=avg_cal_data['total_uncertainty_temp'], yerr=avg_uncal_data['total_uncertainty_resistance'],
                fmt='.', linestyle='-', capsize=1, label='Average Newly Calibrated Data', color='dodgerblue')

    # Plot the comparison data from Paul's calibration
    pauls_calibration_df = pd.read_csv(pauls_calibration_path, sep='\t')

    ax.plot(pauls_calibration_df['Kelvin'], pauls_calibration_df['Resistance'],
            label='Paul\'s Calibration Curve Comparison', color='green', marker='x', linestyle='--', markersize='2')

    # Labels, title, and grid
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12, labelpad=20)
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle='-', alpha=0.5)

    # Legend
    legend = ax.legend(handler_map={plt.Line2D: HandlerLine2D(numpoints=1)})
    for handle in legend.legend_handles:
        handle.set_rasterized(10)

    # Layout adjustments
    plt.tight_layout()
    fig.subplots_adjust(left=0.15)

    # Save and show the plot
    plt.savefig(f"plots\\{title}.png", dpi=1200)
    plt.show()

def alt_alt_new_log_log_plot_calibration_curve_with_uncertainty(avg_cal_data, avg_uncal_data, xlabel, ylabel, title,
                                                                pauls_calibration_path):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Log-transform the data
    log_cal_temp = np.log10(avg_cal_data['Calculated Temp (K)'])
    log_cal_resistance = np.log10(avg_cal_data['Average_Resistance'])
    log_uncal_temp = np.log10(avg_uncal_data['Newly Calculated Temp (K)'])
    log_uncal_resistance = np.log10(avg_uncal_data['Average_Resistance'])

    # Log-transform the uncertainties (assuming they are proportional)
    log_cal_temp_uncertainty = avg_cal_data['total_uncertainty_temp'] / (
                avg_cal_data['Calculated Temp (K)'] * np.log(10))
    log_cal_resistance_uncertainty = avg_cal_data['total_uncertainty_resistance'] / (
                avg_cal_data['Average_Resistance'] * np.log(10))
    log_uncal_temp_uncertainty = avg_uncal_data['total_uncertainty_temp'] / (
                avg_uncal_data['Newly Calculated Temp (K)'] * np.log(10))
    log_uncal_resistance_uncertainty = avg_uncal_data['total_uncertainty_resistance'] / (
                avg_uncal_data['Average_Resistance'] * np.log(10))

    # Plot the log-transformed calibrated data
    ax.errorbar(log_cal_temp, log_cal_resistance,
                xerr=log_cal_temp_uncertainty, yerr=log_cal_resistance_uncertainty,
                fmt='.', linestyle='-', capsize=1, label='Average Calibrated Data', color='red', ecolor='orange',
                markersize='0.5')

    # Plot the log-transformed newly calibrated data with uncertainties
    ax.errorbar(log_uncal_temp, log_uncal_resistance,
                xerr=log_uncal_temp_uncertainty, yerr=log_uncal_resistance_uncertainty,
                fmt='.', linestyle='-', capsize=1, label='Average Newly Calibrated Data', color='dodgerblue',
                ecolor='purple', markersize='0.5')

    # Plot the comparison data from Paul's calibration
    pauls_calibration_df = pd.read_csv(pauls_calibration_path, sep='\t')
    log_pauls_temp = np.log10(pauls_calibration_df['Kelvin'])
    log_pauls_resistance = np.log10(pauls_calibration_df['Resistance'])

    ax.plot(log_pauls_temp, log_pauls_resistance,
            label='Alternative Calibration Curve Comparison', color='green', marker='x', linestyle='--', markersize='2')

    # Labels, title, and grid
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12, labelpad=20)
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle='-', alpha=0.5)

    # Legend
    legend = ax.legend(handler_map={plt.Line2D: HandlerLine2D(numpoints=1)})
    for handle in legend.legend_handles:
        handle.set_rasterized(10)

    # Layout adjustments
    plt.tight_layout()
    fig.subplots_adjust(left=0.15)

    # Save and show the plot
    plt.savefig(f"plots\\{title}.png", dpi=1200)
    plt.show()


def exporting_calibration_curve(avg_uncal_data):
    # Create a new DataFrame with the specified columns
    complete_calibration_curve_df = pd.DataFrame({
        'Temperature (K)': avg_uncal_data['Newly Calculated Temp (K)'],
        'Resistance': avg_uncal_data['Average_Resistance']
    })

    # Save the DataFrame as a CSV file
    complete_calibration_curve_df.to_csv('complete_calibration_curve.csv', index=False)


# Uncertainty from the Chebyshev polynomial application based on temperature range
def chebyshev_rms_error(temp):
    if 0.01 <= temp <= 0.113:
        return 0.00016  # in K
    elif 0.112 < temp <= 1.20:
        return 0.00047  # in K
    elif 1.20 < temp <= 8.01:
        return 0.00095  # in K
    elif 8.00 < temp <= 40.0:
        return 0.00573  # in K
    else:
        return 0.0  # for temperatures outside these ranges

def calculate_uncal_uncertainties_temp(df_cal, df_uncal, popt, std_dev_residuals):
    # Uncertainty from the calibrated sensor (assume constant or vary with temperature as needed)
    df_uncal['uncertainty_cal_sensor'] = df_uncal['Newly Calculated Temp (K)'].apply(
        lambda t: 0.004 if t < 4.2 else 0.01 if t < 10 else 0.035 if t < 20 else 0.076
    )

    # Uncertainty from the polynomial fit (use the standard deviation of the residuals)
    df_uncal['uncertainty_polynomial_fit'] = std_dev_residuals

    # Calculate standard deviation of resistance values
    resistance_std = df_uncal['Average_Resistance'].std()
    # Calculate uncertainty mean based on the number of measurements
    df_uncal['uncertainty_mean'] = resistance_std / np.sqrt(df_uncal['Resistance_Count'])

    # Uncertainty from the Chebyshev polynomial application
    df_uncal['uncertainty_chebyshev'] = df_uncal['Newly Calculated Temp (K)'].apply(chebyshev_rms_error)

    # Total uncertainty propagation (root sum of squares)
    # df_uncal['total_uncertainty_temp'] = np.sqrt(
    #     df_uncal['uncertainty_cal_sensor']**2 +
    #     df_uncal['uncertainty_polynomial_fit']**2 +
    #     df_uncal['uncertainty_mean']**2 +
    #     df_uncal['uncertainty_chebyshev']**2
    # )

    df_uncal['total_uncertainty_temp'] = np.sqrt(
        df_uncal['uncertainty_chebyshev']**2
    )

    return df_uncal


def calculate_cal_uncertainties_temp(df_cal, df_uncal, popt, std_dev_residuals):
    # Uncertainty from the calibrated sensor (assume constant or vary with temperature as needed)
    df_cal['uncertainty_cal_sensor'] = df_cal['Calculated Temp (K)'].apply(
        lambda t: 0.004 if t < 4.2 else 0.01 if t < 10 else 0.035 if t < 20 else 0.076
        # From documentation
    )

    # Uncertainty from the polynomial fit (use the standard deviation of the residuals)
    df_cal['uncertainty_polynomial_fit'] = std_dev_residuals

    # Calculate standard deviation of resistance values
    resistance_std = df_cal['Average_Resistance'].std()
    # Calculate uncertainty mean based on the number of measurements
    df_cal['uncertainty_mean'] = resistance_std / np.sqrt(df_cal['Resistance_Count'])

    # Uncertainty from the Chebyshev polynomial application
    df_cal['uncertainty_chebyshev'] = df_cal['Calculated Temp (K)'].apply(chebyshev_rms_error)

    # # Total uncertainty propagation (root sum of squares)
    # df_cal['total_uncertainty_temp'] = np.sqrt(
    #     df_cal['uncertainty_cal_sensor']**2 +
    #     df_cal['uncertainty_polynomial_fit']**2 +
    #     df_cal['uncertainty_mean']**2 +
    #     df_cal['uncertainty_chebyshev']**2
    # )

    # Total uncertainty propagation (root sum of squares)
    df_cal['total_uncertainty_temp'] = np.sqrt(
        df_cal['uncertainty_chebyshev']**2
    )

    return df_cal

def calculate_uncal_uncertainties_resistance(df_cal, df_uncal, popt, std_dev_residuals):
    # Uncertainty from the calibrated sensor (assume constant or vary with temperature as needed)
    df_uncal['uncertainty_cal_sensor'] = df_uncal['Newly Calculated Temp (K)'].apply(
        lambda t: 0.004 if t < 4.2 else 0.01 if t < 10 else 0.035 if t < 20 else 0.076
    )

    # Uncertainty from the polynomial fit (use the standard deviation of the residuals)
    df_uncal['uncertainty_polynomial_fit'] = std_dev_residuals

    # Calculate standard deviation of resistance values
    resistance_std = df_uncal['Average_Resistance'].std()

    # Calculate uncertainty mean based on the number of measurements
    df_uncal['uncertainty_mean'] = resistance_std / np.sqrt(df_uncal['Resistance_Count'])

    # Uncertainty from the Chebyshev polynomial application
    # no Chebyshev errors for resistance value

    # Total uncertainty propagation (root sum of squares)
    df_uncal['total_uncertainty_resistance'] = np.sqrt(
        df_uncal['uncertainty_cal_sensor']**2 +
        df_uncal['uncertainty_polynomial_fit']**2 +
        df_uncal['uncertainty_mean']**2
    )

    return df_uncal


def calculate_cal_uncertainties_resistance(df_cal, df_uncal, popt, std_dev_residuals):
    # Uncertainty from the calibrated sensor (assume constant or vary with temperature as needed)
    df_cal['uncertainty_cal_sensor'] = df_cal['Calculated Temp (K)'].apply(
        lambda t: 0.004 if t < 4.2 else 0.01 if t < 10 else 0.035 if t < 20 else 0.076
    )

    # Uncertainty from the polynomial fit (use the standard deviation of the residuals)
    df_cal['uncertainty_polynomial_fit'] = std_dev_residuals

    # Calculate standard deviation of resistance values
    resistance_std = df_cal['Average_Resistance'].std()
    # Calculate uncertainty mean based on the number of measurements
    df_cal['uncertainty_mean'] = resistance_std / np.sqrt(df_cal['Resistance_Count'])

    # Uncertainty from the Chebyshev polynomial application
    # no Chebyshev errors for resistance value

    # Total uncertainty propagation (root sum of squares)
    df_cal['total_uncertainty_resistance'] = np.sqrt(
        df_cal['uncertainty_cal_sensor']**2 +
        df_cal['uncertainty_polynomial_fit']**2 +
        df_cal['uncertainty_mean']**2
    )

    return df_cal