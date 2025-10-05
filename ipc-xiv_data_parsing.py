"""Script for parsing data logged at the 2025 IPC."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob




def read_hukseflux(filename):
    with open(filename) as fbuf:
        _ = fbuf.readline()  # skip date line
        _ = fbuf.readline()  # skip empty line
        _ = fbuf.readline()  # skip header line
        # currently only works for one sensor
        sensor = fbuf.readline()
        _ = fbuf.readline()  # skip empty line
        sensor, serial_number = sensor.strip()[1:-1].split(' ')
        df = pd.read_csv(fbuf, sep=',', index_col='Time stamp')
        df.index = pd.to_datetime(df.index, format='%d-%m-%Y %H:%M:%S.%f')
        df.index = df.index.round('1s')
        df = df.asfreq('1s')
        return df


def read_kipp_zonen(filename):
    """
    Read measurement data logged with Kipp & Zonens SmartExplorer software.

    Parameters
    ----------
    filename : path-like
        File name.

    Returns
    -------
    df : pandas.DataFrame
        Measurement data with the index corresponding to the time.

    """
    with open(filename) as fbuf:
        # Skip unused lines
        for _ in range(7):
            fbuf.readline()
        # Read metadata
        meta_columns = fbuf.readline().strip().split(';')
        meta_lines = []
        for _ in range(10):
            meta_lines.append(fbuf.readline().strip().split(';'))
        meta = pd.DataFrame(columns=meta_columns, data=meta_lines)
        meta['Channel'] = meta['Channel'].astype(int)
        meta['Serial Nr'] = meta['Serial Nr'].str.replace('-', '')
        active_channels = meta.loc[meta['Status'] == 'Ready', 'Channel'].to_list()

        fbuf.readline()  # skip empty line

        # Format header
        column_channels = fbuf.readline().strip().split(';')
        column_channels_int = [int(c[3:]) if c[3:].isdigit() else np.nan for c in column_channels]
        column_quantity = fbuf.readline().strip().split(';')
        column_unit = fbuf.readline().strip().split(';')
        usecols = [ii for ii, ch in enumerate(column_channels[3:-1], start=3) if int(ch[3:]) in active_channels]
        header = ['Date', 'Time'] + [f"{meta.loc[meta['Channel']==column_channels_int[ii], 'Serial Nr'].values[0]}-{column_quantity[ii].lower()}" for ii in usecols]
        # Read data
        df = pd.read_csv(fbuf, sep=';', usecols=[1, 2] + usecols, names=header)
        # Format dataframe
        df.index = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y-%m-%d %H:%M:%S')
        df = df.drop(columns=['Date', 'Time'])
        df = df.asfreq('1s')

        return df


# %%

data_path = 'C:/GitHub/IPC-XIV-DTU/'

filenames = glob.glob(data_path + 'Data/2025100*/LOG*.csv')

dfs = [read_kipp_zonen(f) for f in filenames]

df = pd.concat(dfs, axis='rows')
df = df.asfreq('1s')  # need to fill gaps between datafiles

df = df.rename(columns={
    '205241-radiation': 'SHP1-205241',
    '185163-radiation': 'SHP1-185163',
})

# Remove period as instruments were first cleaned here
# Cleaning spike visible in data
df['2025-10-02 07:10': '2025-10-02 07:15'] = np.nan
# Remove period due to infreuent logging (maybe loose connection)
df['2025-10-02 08:10': '2025-10-02 08:45'] = np.nan
# Change calibration coefficient from 8.61577 to 8.52 at 9:42
df.loc['2025-10-02 07:10:00': '2025-10-02 07:42:30', 'SHP1-185163'] = 8.61577 / 8.52 * df.loc['2025-10-02 07:10:00': '2025-10-02 07:42:30', 'SHP1-185163']
#14:26 Both instruments are aligned and cleaned. Made minor adjustments to the first SHP1 too (but very minor).
df.loc['2025-10-02 12:23:00': '2025-10-02 12:27:00', 'SHP1-185163'] = np.nan
df.loc['2025-10-02 12:14:00': '2025-10-02 12:27:00', 'SHP1-205241'] = np.nan
# 15:51: Cleaned both glasses (looked clean already, only tiny speckles)
df.loc['2025-10-02 13:52:00': '2025-10-02 13:52:59', ['SHP1-185163', 'SHP1-205241']] = np.nan

# Seems like an outlier (maybe short accidential shading)
df.loc['2025-10-02 10:49:10', 'SHP1-185163'] = np.nan

# Two shorts dips in DNIs (not present in the other pyrheliometer)
df.loc['2025-10-02 12:18:35': '2025-10-02 12:18:55', 'SHP1-205241'] = np.nan

df['SHP1-205241-adjusted'] = df['SHP1-205241']*1.014

radiation = [c for c in df.columns if c.startswith('SHP1')]

df['2025-10-03 06':].plot(subplots=True, sharex=True, figsize=(10, 10))
#df.loc['2025-10-02 12:05':'2025-10-02 12:28'].plot(subplots=True, sharex=True, figsize=(10, 10), style='.')
#df.loc['2025-10-02 11:05':'2025-10-02 12:45', radiation].plot(style='.')

# %%

df.loc['2025-10-03 06':, radiation].plot(sharex=True, ylim=[850, 1050])

# %%

df['diff'] = df['SHP1-185163'] - df['SHP1-205241']
df['diff-adjusted'] = df['SHP1-185163'] - df['SHP1-205241-adjusted']

df.loc['2025-10-03 06':, 'diff-adjusted'].plot(grid=True, ylim=[-5, 5])

# %%

df.loc['2025-10-03 06':, ['SHP1-185163', 'SHP1-205241']].diff().plot(
    subplots=True, sharex=True, ylim=[-10, 10])


# %%

df.loc['2025-10-02 12:30':, radiation].plot()

df['radiation_ratio'] = df['SHP1-205241'] / df['SHP1-185163']

plt.figure()
df.loc['2025-10-02 12:30':, 'radiation_ratio'].plot(ylim=[0.96, 1.04], grid=True)


# %% Export SHP1 data


for pyrheliometer in ['SHP1-185163', 'SHP1-205241']:
    for date in df.index.to_series().dt.date.unique():
        export = [pyrheliometer, 'WRR 1.000000']
        df_sub = df[df[pyrheliometer].notna() & (df.index.date == date)]
        # !!!Logging time is in UTC time for the Kipp & Zone SmartExplorer software!!!
        df_sub.index = df_sub.index + pd.Timedelta(hours=1)
        df_lines = df_sub.index.strftime('%Y %m %d %H:%M:%S') + ' ' + df_sub[pyrheliometer].astype(str)
        export = export + df_lines.to_list()
        df_export = pd.Series(export)

        filename = f"{pyrheliometer}_{df_sub.index[0].strftime('%y-%m-%d_%H%M')}.dat"
        df_export.to_csv(data_path + 'Data export/' + filename, index=False, header=None)



# %% Hukseflux data
filenames = glob.glob(data_path + 'Data/2025100*/Hukseflux*.csv')

dfs = [read_hukseflux(f) for f in filenames]
df = pd.concat(dfs, axis='rows')

df = df.rename(columns={
    'Irradiance [W/m^2]': 'DR30D1-65086'})

# Spike low in the data
df.loc['2025-10-02 16:35:00': '2025-10-02 16:35:15', 'DR30D1-65086'] = np.nan

df.plot(sharex=True, subplots=True, figsize=(10, 10))

# %% Export DR30 data

pyrheliometer = 'DR30D1-65086'

for date in df.index.to_series().dt.date.unique():
    export = [pyrheliometer, 'WRR 1.000000']
    df_sub = df[df[pyrheliometer].notna() & (df.index.date == date)]
    # !!!Logging time is in local time for the Hukseflux software!!!
    df_sub.index = df_sub.index - pd.Timedelta(hours=1)
    df_lines = df_sub.index.strftime('%Y %m %d %H:%M:%S') + ' ' + df_sub[pyrheliometer].astype(str)
    export = export + df_lines.to_list()
    df_export = pd.Series(export)

    filename = f"{pyrheliometer}_{df_sub.index[0].strftime('%y-%m-%d_%H%M')}.dat"
    df_export.to_csv(data_path + 'Data export/' + filename, index=False, header=None)





# %%

import minimalmodbus
PORT = 'COM9'
MODBUS_ADDRESS = 2
#Set up instrument
instrument = minimalmodbus.Instrument(PORT,MODBUS_ADDRESS,mode=minimalmodbus.MODE_RTU)

#Make the settings explicit
instrument.serial.baudrate = 19200        # Baud
instrument.serial.bytesize = 8
instrument.serial.parity   = minimalmodbus.serial.PARITY_EVEN
instrument.serial.stopbits = 1
instrument.serial.timeout  = 1          # seconds

# Good practice
instrument.close_port_after_each_call = True
instrument.clear_buffers_before_each_transaction = True


REGISTER = 8#2

# Read temperatureas a float
# if you need to read a 16 bit register use instrument.read_register()

output = instrument.read_float(REGISTER)

# Temperature compensated radiation in W/m2
# The raw data from the sensor is calibrated, linearized; temperature compensated and filtered
sensor_1_data_corrected = instrument.read_register(5)
sensor_1_data_uncorrected = instrument.read_register(6)

# The body temperature sensor measures the temperature of the body in 0.1°C.
body_temperature = instrument.read_register(8) / 10
# The Ext power sensor measured the external voltage applied to the sensor in 0.1 Volt
ext_power_sensor = instrument.read_register(9) / 10


print(ext_power_sensor)

# %%

