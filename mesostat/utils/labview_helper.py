import pandas as pd

from mesostat.utils.arrays import string_list_pad_unique


def _parse_event_df(dfTrial, timestampGlobal, iTrial):
    # Make sure all fields are unique
    eventList = list(dfTrial['Event'])
    if len(dfTrial) != len(set(eventList)):
        dfTrial['Event'] = string_list_pad_unique(eventList, suffix='')

    # Process times
    timestampTrial = min(list(dfTrial['Datetime']))
    timestampTrialSeconds = (timestampTrial - timestampGlobal).total_seconds()
    rowsDict = {row['Event']: (row['Datetime'] - timestampTrial).total_seconds() for idx, row in dfTrial.iterrows()}

    return pd.DataFrame({**{'Trial': iTrial, 'Time': timestampTrialSeconds}, **rowsDict}, index=[0])


# Read labview file of structure ['Date', 'Time', 'Trial', 'Event']
# Convert each trial into row of dataframe
# For each row, return trial index and time in seconds since start of a session
# For each event in a row, return time since start of the trial, or NaN if event did not occur
#
def labview_parse_log_as_pandas(pwd, sep='\t', endEvent=None):
    dfRAW = pd.read_csv(pwd, sep=sep)

    if list(dfRAW.columns) != ['Date', 'Time', 'Trial', 'Event']:
        raise IOError("Only work with specific structure, have " + str(list(dfRAW.columns)))

    # Strip any whitespace
    dfRAW['Event'] = dfRAW['Event'].str.strip()

    # Parse DateTime as object
    dfRAW['Datetime'] = dfRAW['Date'] + dfRAW['Time']
    dfRAW['Datetime'] = pd.to_datetime(dfRAW['Datetime'], infer_datetime_format=True)
    dfRAW = dfRAW.drop(['Date', 'Time'], axis=1)

    # Parse trials
    dfRAW['Trial'] = dfRAW['Trial'].astype(int)

    # Find the start time
    timestampGlobal = min(list(dfRAW['Datetime']))
    print(timestampGlobal)

    # Split data by trials
    dfRez = pd.DataFrame()
    if endEvent is None:
        # Split trials using trial indices
        trialIdxs = sorted(set(dfRAW['Trial']))
        for iTrial in trialIdxs:
            dfTrial = dfRAW[dfRAW['Trial'] == iTrial].copy()

            dfTrialFlat = _parse_event_df(dfTrial, timestampGlobal, iTrial)
            dfRez = dfRez.append(dfTrialFlat)
    else:
        # Split trials using end-trial events
        dfTrial = pd.DataFrame()
        iTrial = 1
        for idx, row in dfRAW.iterrows():
            dfTrial = dfTrial.append(row)
            if row['Event'] == endEvent:
                dfTrialFlat = _parse_event_df(dfTrial, timestampGlobal, iTrial)
                dfRez = dfRez.append(dfTrialFlat)

                iTrial += 1
                dfTrial = pd.DataFrame()

    # pd.set_option("max_columns", None)
    return timestampGlobal, dfRez
