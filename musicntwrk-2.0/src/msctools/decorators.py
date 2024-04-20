#
# msctools: my collection of composing and performing tools in python
#
# © 2023 Marco Buongiorno Nardelli
#

import threading
from functools import wraps

def threading_decorator(func):
    @wraps(func)
    def wrapper(*args):
        result = threading.Thread(target=func,args=args).start()
        return result
    return wrapper