"""
A seven-segment number display module.
"""

def getSevSegStr(number, minWidth=0):
    """
    Return a seven-segment display string of number.
    The returned string will be padded with zeros if it is smaller than minWidth.
    """

    number = str(number).zfill(minWidth)

    rows = ['', '', '']

    for i, numeral in enumerate(number):

        if numeral == '.':
            rows[0] += ' '
            rows[1] += ' '
            rows[2] += '.'
            continue
        elif numeral == '-':
            rows[0] += '    '
            rows[1] += ' __ '
            rows[2] += '    '
        elif numeral == '0':
            rows[0] += ' __ '
            rows[1] += '|  |'
            rows[2] += '|__|'
        elif numeral == '1':
            rows[0] += '    '
            rows[1] += '   |'
            rows[2] += '   |'
        elif numeral == '2':
            rows[0] += ' __ '
            rows[1] += ' __|'
            rows[2] += '|__ '
        elif numeral == '3':
            rows[0] += ' __ '
            rows[1] += ' __|'
            rows[2] += ' __|'
        elif numeral == '4':
            rows[0] += '    '
            rows[1] += '|__|'
            rows[2] += '   |'
        elif numeral == '5':
            rows[0] += ' __ '
            rows[1] += '|__ '
            rows[2] += ' __|'
        elif numeral == '6':
            rows[0] += ' __ '
            rows[1] += '|__ '
            rows[2] += '|__|'
        elif numeral == '7':
            rows[0] += ' __ '
            rows[1] += '   |'
            rows[2] += '   |'
        elif numeral == '8':
            rows[0] += ' __ '
            rows[1] += '|__|'
            rows[2] += '|__|'
        elif numeral == '9':
            rows[0] += ' __ '
            rows[1] += '|__|'
            rows[2] += ' __|'

        if i != len(number) - 1 and number[i + 1] != '.':
            rows[0] += ' '
            rows[1] += ' '
            rows[2] += ' '

    return '\n'.join(rows)

# If the program isn't being imported, display the numbers 00 to 99.
if __name__ == '__main__':
    print("This module is meant to be imported rather than run.")