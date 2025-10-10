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

data_path = 'C:/GitHub/IPC-XIV-DTU/Data/'

filenames = glob.glob(data_path + '2025*/LOG*.csv')

dfs = [read_kipp_zonen(f) for f in filenames]

df = pd.concat(dfs, axis='rows')
df = df.asfreq('1s')  # need to fill gaps between datafiles

df = df.rename(columns={
    '205241-radiation': 'SHP1-205241',
    '185163-radiation': 'SHP1-185163-raw',
})

df['SHP1-185163'] = df['SHP1-185163-raw'] * 8.52 / 8.518

df['SHP1-205241-wrr'] = df['SHP1-205241']*1.01
df['SHP1-185163-wrr'] = df['SHP1-185163']*0.997

radiation = [c for c in df.columns if c.startswith('SHP1')]
radiation_wrr = [c for c in df.columns if c.endswith('wrr')]

(df['SHP1-185163-wrr'] / df['SHP1-205241-wrr']).plot(
    ylim=[0.96, 1.04], ylabel='SHP1 ratio 185163/205241', grid=True)

# %% Data removal

if '2025-10-10' in df.index.strftime('%Y-%m-%d'):
    df.loc['2025-10-10 07 06:58': '2025-10-10 07:23:20', 'SHP1-185163'] = np.nan

if '2025-10-02' in df.index.strftime('%Y-%m-%d'):
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


# %%

df['diff'] = df['SHP1-185163'] - df['SHP1-205241']
df['diff-adjusted'] = df['SHP1-185163'] - df['SHP1-205241-adjusted']

df.loc[:, 'diff-adjusted'].plot(grid=True, ylim=[-5, 5])

# %%

df.loc['2025-10-03 06':, ['SHP1-185163', 'SHP1-205241']].diff().plot(
    subplots=True, sharex=True, ylim=[-10, 10])


# %%

df.loc['2025-10-02 12:30':, radiation].plot()

df['radiation_ratio'] = df['SHP1-205241-adjusted'] / df['SHP1-185163']

plt.figure()
df.loc[:, 'radiation_ratio'].plot(ylim=[0.99, 1.01], grid=True)


# %% Export SHP1 data


for pyrheliometer in ['SHP1-185163', 'SHP1-205241']:
    for date in df.index.to_series().dt.date.unique():
        export = [pyrheliometer, 'WRR 1.000000']
        df_sub = df[df[pyrheliometer].notna() & (df.index.date == date)]
        if df_sub.empty:
            continue  # skip empty files
        # !!!Logging time is in UTC time for the Kipp & Zone SmartExplorer software!!!
        df_sub.index = df_sub.index + pd.Timedelta(hours=1)
        df_lines = \
            df_sub.index.strftime('%Y %m %d %H:%M:%S') + ' ' + df_sub[pyrheliometer].astype(str)
        export = export + df_lines.to_list()
        df_export = pd.Series(export)

        filename = f"{pyrheliometer}_{df_sub.index[0].strftime('%y-%m-%d_%H%M')}.dat"
        df_export.to_csv(data_path + '../Data export/' + filename, index=False, header=None)


# %% Hukseflux data
filenames = glob.glob(data_path + '202510*/Hukseflux*.csv')

dfs = [read_hukseflux(f) for f in filenames]
df = pd.concat(dfs, axis='rows')

df = df.rename(columns={
    'Irradiance [W/m^2]': 'DR30D1-65086'})

df.plot(sharex=True, subplots=True, figsize=(10, 10))

# Spike low in the data
if '2025-10-02' in df.index.strftime('%Y-%m-%d'):
    df.loc['2025-10-02 16:35:00': '2025-10-02 16:35:15', 'DR30D1-65086'] = np.nan

# Cleaning (low spike)
if '2025-10-10' in df.index.strftime('%Y-%m-%d'):
    df.loc['2025-10-10 16:13:35': '2025-10-10 16:13:50', 'DR30D1-65086'] = np.nan
    # Interpolation of missing time stamp
    df.loc['2025-10-10 16:18:00', 'DR30D1-65086'] = (862.91 + 862.93) / 2


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
    df_export.to_csv(data_path + '../Data export/' + filename, index=False, header=None)


# %% CHP1 pyrheliometer (from Georgia's logging system)

filenames = glob.glob(data_path + '202510*/IPC_DNI_2025*.dat')

dfs = []
for f in filenames:
    dfi = pd.read_csv(f, skiprows=[0, 2, 3], index_col=[0])
    dfi.index = pd.to_datetime(dfi.index, format='%Y-%m-%d %H:%M:%S')
    dfs.append(dfi)

df = pd.concat(dfs, axis='rows')

df = df.rename(columns={'Vraw_mV_Adam': 'CHP1-090137'})

if '2025-10-08' in df.index.strftime('%Y-%m-%d'):
    df['2025-10-08 10:28:30': '2025-10-08 10:33:59'] = np.nan
    df['2025-10-08 15:19:30': '2025-10-08 15:25:30'] = np.nan
df = df[['CHP1-090137']]
df['CHP1-090137'] = df['CHP1-090137'] / 8.08
df.plot()

# %%
pyrheliometer = 'CHP1-090137'

for date in df.index.to_series().dt.date.unique():
    export = [pyrheliometer, 'WRR 1.000000']
    df_sub = df[df[pyrheliometer].notna() & (df.index.date == date)]
    df_lines = df_sub.index.strftime('%Y %m %d %H:%M:%S') + ' ' + df_sub[pyrheliometer].astype(str)
    export = export + df_lines.to_list()
    df_export = pd.Series(export)

    filename = f"{pyrheliometer}_{df_sub.index[0].strftime('%y-%m-%d_%H%M')}.dat"
    df_export.to_csv(data_path + '../Data export/' + filename, index=False, header=None)

# %%
