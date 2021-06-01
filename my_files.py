def get_text(file):
    """Returns text of input file path as string"""
    with open(file, 'r') as file:
        text = file.read()
    return text

#########################################################################################