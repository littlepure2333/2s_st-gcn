def change_font_color(original_str,color="red"):
    '''Change the color of the font to be printed in terminal.
    The color can be changed to black, red, green, yellow, blue, purple, cyan and white.
    The default color is red.
    '''
    assert type(original_str) is str, "not a string"
    
    if color == "black":
        new_str = "\033[30;40m" + original_str + "\033[0m"
    elif color == "red":
        new_str = "\033[31;40m" + original_str + "\033[0m"
    elif color == "green":
        new_str = "\033[32;40m" + original_str + "\033[0m"
    elif color == "yellow":
        new_str = "\033[33;40m" + original_str + "\033[0m"
    elif color == "blue":
        new_str = "\033[34;40m" + original_str + "\033[0m"
    elif color == "purple":
        new_str = "\033[35;40m" + original_str + "\033[0m"
    elif color == "cyan":
        new_str = "\033[36;40m" + original_str + "\033[0m"
    else: # white and other invalid parameter
        new_str = original_str

    return new_str