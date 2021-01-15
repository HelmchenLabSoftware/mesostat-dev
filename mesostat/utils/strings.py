import re


def path_extract_data(s, template, charL='<', charR='>', charDir='/'):
    nTemplate = len(template.split(charDir))
    sTrunc = charDir.join(s.split(charDir)[-nTemplate:])
    return str_extract_data(sTrunc, template, charL=charL, charR=charR)


def str_extract_data(string, template, charL='<', charR='>'):
    '''

    :param string:    String to be parsed
    :param template:  Template denoting which parts to parse and how to call them
    :param charL:     Character denoting start of parsed block
    :param charR:     Character denoting end of parsed block
    :return:          Dictionary of parsed values

    Example:
       >>> str_extract_data('testing_cat/12/data.txt', 'testing_<animal>/<age>/data.txt')
       {'animal': 'cat', 'age': '12'}
    '''

    # Find keys that need to be extracted
    keys = re.findall('\\' + charL + '.*?\\' + charR, template)

    # Convert template to regex standard
    tTmp = template
    for k in keys:
        tTmp = tTmp.replace(k, '(.*?)')

    # Extract
    m = re.match(tTmp, string)                         # Extract data from string
    keys = [k[len(charL):-len(charR)] for k in keys]   # Crop the tags from the keys

    # Return dictionary
    return {keys[i] : m.group(i+1) for i in range(len(keys))}
