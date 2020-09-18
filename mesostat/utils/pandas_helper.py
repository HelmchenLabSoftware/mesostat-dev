import pandas as pd
from mesostat.utils.algorithms import non_uniform_base_arithmetic_iterator


def get_one_row(rows):
    nRows = rows.shape[0]
    if nRows == 0:
        return None, None
    elif nRows > 1:
        raise ValueError("Expected 1 match, got", nRows)

    for idx, row in rows.iterrows():
        return idx, row


def get_rows_colval(df, colname, val):
    return df[df[colname] == val]


# Get rows for which several columns have some exact values
def get_rows_colvals(df, queryDict, dropQuery=False):
    if len(queryDict) == 0:
        return df
    else:
        # Query likes strings to be wrapped in quotation marks for later evaluation
        strwrap = lambda val: '"' + val + '"' if isinstance(val, str) else str(val)
        query = ' and '.join([colname+'=='+strwrap(val) for colname, val in queryDict.items()])
        rez = df.query(query)
        if dropQuery:
            return rez.drop(columns=list(queryDict.keys()))
        else:
            return rez


# Return all rows for which values match the list exactly
def get_rows_colvals_exact(df, lst):
    return get_rows_colvals(df, dict(zip(df.columns, lst)))


# Check if there is at least 1 row that matches the list exactly
def row_exists(df, lst):
    return len(get_rows_colvals_exact(df, lst)) > 0


# Add new row to dataframe, unless such a row is already present
def add_list_as_row(df, lst, skip_repeat=True):
    if skip_repeat and row_exists(df, lst):
        print("Skipping existing row", lst)
        return df
    else:
        newRow = pd.DataFrame([lst], columns=df.columns)
        return df.append(newRow, ignore_index=True)


# Get a dictionary where keys are column names and values are possible values for that column
# Construct a dataframe where rows are all combinations of provided column values
def outer_product_df(d):
    rowsLst = []
    baseArr = [len(v) for v in d.values()]
    sweepIter = non_uniform_base_arithmetic_iterator(baseArr)

    for valIdxs in sweepIter:
        rowsLst += [[v[i] for v, i in zip(d.values(), valIdxs)]]

    return pd.DataFrame(rowsLst, columns = list(d.keys()))


def merge_df_from_dict(dfDict, columnNames):
    '''
    :param dfDict: keys are extra column values as tuple. Values are dataframes. All dataframes must have same columns
    :param columnNames: names of the extra columns
    :return: a single dataframe that merges other dataframes using extra columns
    '''

    rezDFList = []
    for k, v in dfDict.items():
        dfCopy = v.copy()
        # Iterate in reverse order because we will be inserting each column at the beginning
        for colname, colval in zip(columnNames[::-1], k[::-1]):
            dfCopy.insert(0, colname, colval)

        rezDFList += [dfCopy]

    return pd.concat(rezDFList, sort=False).reset_index(drop=True)