#!/usr/bin/python3
import pandas as pd
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as ax
from matplotlib.dates import DateFormatter
from matplotlib.dates import MinuteLocator, DateFormatter
from scipy.interpolate import interp1d
from matplotlib.ticker import ScalarFormatter
import re
#mpl.use('Qt5Agg')


from post_processing_data_analysis_functions_va_7 import *

if __name__ == '__main__':

    # Path to the CSV files
    file_paths = 'C:\\Users\\PHR24AS\\PycharmProjects\\pythonProject1\\Run-09-2024-07-01-UP'

    ###
    ROx_coeff_file_path = 'C://Users//PHR24AS//PycharmProjects//pythonProject1//Calibration//U10391//U10391.cof'
    ROx_data_file_path = 'C://Users//PHR24AS//PycharmProjects//pythonProject1//Calibration//U10391//U10391.dat'

    Cernox_coeff_file_path = 'C://Users//PHR24AS//PycharmProjects//pythonProject1//Calibration//x200595-Cernox-4K//X200595.cof'
    Cernox_data_file_path = 'C://Users//PHR24AS//PycharmProjects//pythonProject1//Calibration//x200595-Cernox-4K//X200595.dat'
    ###
    #Pauls Calibration curve to compare
    #calibration_curve_paul_path = 'C://Users/Annora Sundararajan//PycharmProjects//pythonProject1//U10626CalibrationCurve_paul'
    ###

    mapping = {1: "(Cernox - Calibrated)",
               2: "(U10391 - Calibrated)",
               3: "(U10613 - UnCal)",
               5: "(PT100 - UnCal)",
               10:"(U10626 - UnCal)"}

    print('Reading has begun')


    combined_df = load_csv_files(file_paths)

    label_channels(combined_df,mapping)

    #process_Cernox_data(Cernox_data_file_path, Cernox_coeff_file_path)

    #process_ROx_data(ROx_data_file_path, ROx_coeff_file_path)

    combined_df = covert_datetime(combined_df)

    print('Processing has begun')


    process_ROx_fridge_data_chebychev(ROx_coeff_file_path, combined_df)

    sensor_name_dfs = split_by_sensor_name(combined_df)

    setpoints, setpoints_df = create_setpoints_list(calibration_data_file_path=ROx_data_file_path)


    df_stable_ROx_Calibrated_0_1_std, ROx_Calibrated_0_1_std_ranges_df = identify_stable_regions(
        df=sensor_name_dfs["(U10391 - Calibrated)"],
        window_size=5,
        min_duration=1,
        tolerance_fraction=0.001,
        setpoints_df=setpoints_df)


    filtered_uncal_data = filter_data_by_ranges(df=sensor_name_dfs["(U10626 - UnCal)"],
                                                      ranges_df=ROx_Calibrated_0_1_std_ranges_df,
                                                      timestamp_column="Time Stamp")

    filtered_cal_data = filter_data_by_ranges(df=sensor_name_dfs["(U10391 - Calibrated)"],
                                                      ranges_df=ROx_Calibrated_0_1_std_ranges_df,
                                                      timestamp_column="Time Stamp")

    avg_resistance_uncal_df, filtered_data_1 = filter_and_calculate_averages(df=sensor_name_dfs["(U10626 - UnCal)"],
                                              ranges_df=ROx_Calibrated_0_1_std_ranges_df,
                                              timestamp_column="Time Stamp")

    avg_resistance_cal_df, filtered_data = filter_and_calculate_averages(df=sensor_name_dfs["(U10391 - Calibrated)"],
                                              ranges_df=ROx_Calibrated_0_1_std_ranges_df,
                                              timestamp_column="Time Stamp")

    average_uncal_data = calculate_average_timestamp(df=avg_resistance_uncal_df,
                                                     ts1='Start_Timestamp', ts2='End_Timestamp')

    average_cal_data = calculate_average_timestamp(df=avg_resistance_cal_df,
                                                     ts1='Start_Timestamp', ts2='End_Timestamp')

    calibrated_data = calibrate_data_chebychev(ROx_coeff_file_path, avg_resistance_uncal_df)
    #rename this to calculate temperature or similar

    cal_cal_data = calibrate_data_chebychev(ROx_coeff_file_path, avg_resistance_cal_df)

    print('Plotting has begun')


    #Plot all raw data from all sensors
    plot_xy_columns_from_dfs(sensor_name_dfs,'Time Stamp', 'Resistance',
                             title="Raw data from all sensors", xlabel="Time Stamp", ylabel="Resistance (Ω)")

    #Plot Calibrated data with setpoint stability
    alt_plot_calibrated_ROx_with_stability(df=sensor_name_dfs["(U10391 - Calibrated)"],
                                       stable_df=df_stable_ROx_Calibrated_0_1_std,
                                       x_column='Time Stamp', y_column='Resistance',
                                       title="Calibrated ROx with stability",
                                       xlabel="Time Stamp", ylabel="Resistance (Ω)")

    plot_uncalibrated_ROx(df=sensor_name_dfs["(U10391 - Calibrated)"], filtered_uncal=filtered_uncal_data,
                                       stable_df=df_stable_ROx_Calibrated_0_1_std, avg_uncal_data=average_uncal_data,
                                       x_column='Time Stamp', y_column='Resistance',
                                       title="Filtered Uncalibrated ROx Sensor U10626  ",
                                       xlabel="Time Stamp", ylabel="Resistance (Ω)")

    plot_avg_resistance(avg_resistance_cal_df, avg_resistance_uncal_df, xlabel='average_timestamp',
                        ylabel='Average_Resistance', title='Average Resistance cal, uncal')

    plot_calibration_curve(avg_cal_data=avg_resistance_cal_df,
                           avg_uncal_data=avg_resistance_uncal_df,
                           xlabel='Calculated Temp (K)',
                           ylabel='Average_Resistance',
                           title='Calibration Curve, Calibrated and Uncalibrated')

    # Fit the polynomial model and plot the fitted curve
    popt = fit_and_plot_polynomial(avg_resistance_cal_df, avg_resistance_uncal_df,
                            calibrated_label='Average_Resistance', uncalibrated_label='Average_Resistance')

    # Apply the calibration to get calibrated resistance values
    newly_calibrated_resistances_df = apply_calibration(avg_resistance_uncal_df, uncalibrated_label='Average_Resistance', popt=popt)

    # After the fit, calculate the residuals and std deviation
    avg_resistance_cal_df['residuals'] = avg_resistance_cal_df['fitted_resistance'] - avg_resistance_cal_df[
        'Average_Resistance']
    std_dev_residuals = avg_resistance_cal_df['residuals'].std()


    # Plot the calibration results to show the effectiveness
    plot_calibration_results(avg_resistance_cal_df, newly_calibrated_resistances_df,
                            calibrated_label='Average_Resistance',
                             uncalibrated_label='Average_Resistance',
                             calibrated_res_label='newly_calibrated_resistance',
                             x_label='Uncalibrated Sensor Resistance (Ω)',
                             y_label='Calibrated Sensor Resistance (Ω)',
                             title='Comparison of Original and Calibrated Values')

    # Plot the alt calibration results to show the effectiveness
    alt_plot_calibration_results(avg_resistance_cal_df, newly_calibrated_resistances_df,
                            calibrated_label='Average_Resistance',
                             uncalibrated_label='Average_Resistance',
                             calibrated_res_label='newly_calibrated_resistance',
                             x_label='Originally Uncalibrated Sensor Resistance (Ω)',
                             y_label='Calibrated Resistance (Ω)',
                             title='Alternative Comparison of Original Scatter Plot and Newly Calibrated Values')


    # Calculate the temperature
    newly_calibrated_resistances_df = calculate_temp_via_chebychev(
        ROx_coeff_file_path, newly_calibrated_resistances_df,
        resistance_column_title='newly_calibrated_resistance')

    # Calculate uncertainties for the newly calibrated data
    newly_calibrated_resistances_df = calculate_uncal_uncertainties_temp(
        df_cal=avg_resistance_cal_df,
        df_uncal=newly_calibrated_resistances_df,
        popt=popt,
        std_dev_residuals=std_dev_residuals
    )

    avg_resistance_cal_df = calculate_cal_uncertainties_temp(
        df_cal=avg_resistance_cal_df,
        df_uncal=newly_calibrated_resistances_df,
        popt=popt,
        std_dev_residuals=std_dev_residuals
    )

    newly_calibrated_resistances_df = calculate_uncal_uncertainties_resistance(
        df_cal=avg_resistance_cal_df,
        df_uncal=newly_calibrated_resistances_df,
        popt=popt,
        std_dev_residuals=std_dev_residuals
    )

    avg_resistance_cal_df = calculate_cal_uncertainties_resistance(
        df_cal=avg_resistance_cal_df,
        df_uncal=newly_calibrated_resistances_df,
        popt=popt,
        std_dev_residuals=std_dev_residuals
    )

    # alt_alt_new_log_log_plot_calibration_curve_with_uncertainty(
    #     avg_cal_data=avg_resistance_cal_df,
    #     avg_uncal_data=newly_calibrated_resistances_df,
    #     xlabel='Log base 10 Calculated Temp (K)',
    #     ylabel='Log base 10 Average Resistance',
    #     title='Calibration Curve, with Newly Calibrated Temperatures and Uncertainties (Logarithm Base 10 Plot)',
    #     pauls_calibration_path=calibration_curve_paul_path)

    exporting_calibration_curve(avg_uncal_data=newly_calibrated_resistances_df)

    print('fin')
