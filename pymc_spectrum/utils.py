def display_data(data):
    '''draw a bar chart using text blocks
    
    Parameters
    ----------
    data : array-like
        1d array filled with numbers
    '''

    max_chars = 50
    max_value = max(data)
    for d in data:
        n = int(d * max_chars / max_value)
        s = '█' * n
        if d == 0:
            s = '_'
        print(s + f'     {d:.2f}')
    print(data)
