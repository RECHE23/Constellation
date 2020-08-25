# -*- coding: utf-8 -*-


def display_progress_bar(iteration, total, prefix='', suffix='',
                         decimals=1, length=100, fill=u'\u2588',
                         print_end=u"\r"):
    if iteration == 0:
        print(u'\x1b[?25l', end='')
    percent = ("{0:." + str(decimals) + "f}").format(
        100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    line = u'\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix)
    print(line, end=print_end)
    if iteration == total:
        print(u'\r' + ' ' * 120 + u'\r', end=u'\x1b[?25h')
