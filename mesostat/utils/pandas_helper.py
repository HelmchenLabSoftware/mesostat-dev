import pandas as pd
import itertools


def pd_is_one_row(rows):
    nRows = rows.shape[0]
    if nRows == 0:
        return None, None
    elif nRows > 1:
        raise ValueError("Expected 1 match, got", nRows)

    for idx, row in rows.iterrows():
        return idx, row


def pd_rows_colval(df, colname, val):
    return df[df[colname] == val]


# Get rows for which several columns have some exact values
# FIXME: Does not work with complex datatypes like tuple
# TODO: Implement partial matches
# TODO: Implement inequalities
def pd_query(df, queryDict, dropQuery=False):
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
def pd_query_exact(df, lst):
    return pd_query(df, dict(zip(df.columns, lst)))


# Check if there is at least 1 row that matches the list exactly
def pd_row_exists(df, lst):
    return len(pd_query_exact(df, lst)) > 0


# Add new row to dataframe, unless such a row is already present
def pd_append_row(df, lst, skip_repeat=False):
    if skip_repeat:
        if pd_row_exists(df, lst):
            print("Skipping existing row", lst)
            return df
    else:
        newRow = pd.DataFrame([lst], columns=df.columns)
        return df.append(newRow, ignore_index=True)

# Appends all dataframes in a list
# A new column is added with values unique to the rows of original dataframes
def pd_vstack_df(dfLst, colName, colVals):
    rez = pd.DataFrame()
    for df, val in zip(dfLst, colVals):
        df1 = df.copy()
        df1[colName] = val
        rez = rez.append(df1)
    return rez.reset_index()


# Merge several dataframes with exactly the same structure by adding columns that have different values
# TODO: Test that dataframes are indeed equivalent
# TODO: Test that dataframe values are exactly the same except of split cols
def pd_merge_equivalent_df(dfLst, splitColNames, dfNames):
    dfRez = dfLst[0].copy()
    dfRez = dfRez.drop(splitColNames)
    for df, dfName in zip(dfLst, dfNames):
        for colName in splitColNames:
            dfRez[colName + '_' + dfName] = df[colName]
    return dfRez


# Get a dictionary where keys are column names and values are possible values for that column
# Construct a dataframe where rows are all combinations of provided column values
def outer_product_df(d):
    rowsLst = list(itertools.product(*d.values()))
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


# Move some of the columns in front in that order, the rest stay in the same order at the end
def pd_move_cols_front(df, colsMove):
    colsNew = list(df.keys())
    for col in colsMove[::-1]:
        colsNew.insert(0, colsNew.pop(colsNew.index(col)))

    return df.loc[:, colsNew]
