

import mesostat.utils.pandas_helper as pandas_helper

outerDict = {
    'aaa' : [1, None, 3],
    'bbb' : ['cat', 'dog', 'rat', None]
}

print(pandas_helper.outer_product_df(outerDict))
