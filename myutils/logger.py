



class Logger(object):
    def __init__(self, log_filepath, title=None, resume=False):
        self.resume = resume
        self.title = '' if title is None else title
        self.log_filepath = log_filepath
        self.file = None
        if self.log_filepath is not None:
            if resume:
                # resume from a former logfile
                pass
            else:
                self.file = open(self.log_filepath, 'w')

    def set_names(self, names):
        if self.resume:
            pass
        if self.file is None:
            print('Error: should not call set_names before a Logger is created')
            return
        self.numbers = {}
        self.names = names
        for idx, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()
