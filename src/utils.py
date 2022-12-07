def is_number(s):
    try:
        if type(s)!=str:
            s = str(s)
        float(s)
        return True
    except ValueError:
        return False