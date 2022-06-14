def join_string(strings):
    strings = [str(s) if s is not None else '' for s in strings]

    return ' '.join(strings)

