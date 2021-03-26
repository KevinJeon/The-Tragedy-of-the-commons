from threading import Thread


class ReturnableThread(Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(self._args)

    def join(self, *args):
        Thread.join(self, *args)
        return self._return
