import unittest
import pandas as pd
# from sklearn.datasets import load_boston

import mesostat.utils.pandas_helper as ph



class TestMetricAutocorr(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        data = [
            ['cat', 5, 'F', 'Susan'],
            ['cat', 6, 'M', 'Johnny'],
            ['dog', 10, 'M', 'Bork'],
            ['dog', 12, 'F', 'Fufu'],
            ['rat', 3, 'F', 'Mika']
        ]

        self.df = pd.DataFrame(data, columns=['species', 'weight', 'sex', 'name'])

        # dataDict = load_boston()
        # self.dataExample = pd.DataFrame(dataDict['data'], columns=dataDict['feature_names'])

    def test_first_row(self):
        row1 = ph.pd_first_row(self.df)[1]
        row2 = ph.pd_is_one_row(ph.pd_query(self.df, {'name': 'Susan'}))[1]
        self.assertTrue(row1.equals(row2))

    def test_rows_colval(self):
        df1 = ph.pd_rows_colval(self.df, 'species', 'dog')
        df2 = ph.pd_query(self.df, {'species': 'dog'})
        self.assertTrue(df1.equals(df2))

    def test_pd_query(self):
        # Test no query has no result
        df1 = ph.pd_query(self.df, {})
        self.assertTrue(df1.equals(self.df))

        # Test empty dataframe queries to empty dataframe
        df1 = ph.pd_query(pd.DataFrame(), {'species': 'dog'})
        self.assertEqual(len(df1), 0)

        # Test non-existing column or value
        df1 = ph.pd_query(pd.DataFrame(), {'house': 10})
        df2 = ph.pd_query(pd.DataFrame(), {'species': 'monkey'})
        self.assertEqual(len(df1), 0)
        self.assertEqual(len(df2), 0)

        # Assert multiquery
        df1 = ph.pd_query(self.df, {'species': 'dog', 'sex': 'F'})
        row = ph.pd_is_one_row(df1)[1]
        self.assertEqual(row['name'], 'Fufu')

    def test_row_exists(self):
        isRowGood = ph.pd_row_exists(self.df, ['dog', 10, 'M', 'Bork'])
        isRowBad = ph.pd_row_exists(self.df, ['dog', 10, 'M', 'Johnny'])

        self.assertTrue(isRowGood)
        self.assertFalse(isRowBad)

    def test_append_row(self):
        dfNew = self.df.copy()
        dfNew = ph.pd_append_row(dfNew, ['rat', 4, 'F', 'Gaga'])
        self.assertEqual(len(self.df), 5)
        self.assertEqual(len(dfNew), 6)

    def test_outer_product(self):
        pass
        # outerDict = {
        #     'aaa': [1, None, 3],
        #     'bbb': ['cat', 'dog', 'rat', None]
        # }
        #
        # print(pandas_helper.outer_product_df(outerDict))

unittest.main()
