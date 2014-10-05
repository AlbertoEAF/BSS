#! /usr/bin/python3

class ConfigParser:
    def __init__(self, configpath):
        config = open(configpath, 'r')
        lines = [ line.strip() for line in config.readlines() if line.strip() ]
        config.close()

        self.d = {}

        for line in lines:
            l = line.split('=')
            if len(l) == 2:
                self.d[l[0].strip()] = l[1].strip()
            elif l[0].strip()[0] != "#":
                error("Invalid line on .cfg file: <{}>".format(line))

    def __getitem__(self, key):
        return self.d[key]

    def i(self, key):
        return int(self.d[key])

    def f(self, key):
        return float(self.d[key])
