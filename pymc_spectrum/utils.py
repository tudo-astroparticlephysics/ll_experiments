def display_data(data, mark_indices=None, low_index=None, high_index=None):
    '''draw a bar chart using text blocks
    
    Parameters
    ----------
    data : array-like
        1d array filled with numbers
    '''
    max_chars = 50
    max_value = max(data)
    for i, d in enumerate(data):
        n = int(d * max_chars / max_value)
        if mark_indices is not None:
            if i in mark_indices:
                s = '█' * n
            else:
                s = '*' * n
        else:
            s = '█' * n
        if d == 0:
            s = '_'
        if low_index:
            if i == low_index:
                s += '-' * max_chars

        if high_index:
            if i == high_index:
                s += '-' * max_chars

        print(s + f'     {d:.2f}')
    # print(data)
