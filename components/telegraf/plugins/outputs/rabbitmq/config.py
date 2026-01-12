from pathlib import Path
import configparser
import os

# path to config file
pathToConfig = Path(__file__).parent / 'config.ini'

# path to config file with overrides when environment variable DEVMODE is set to 1
pathToConfigOverrides = Path('dev/configoverrides.ini')


class CustomConfig():
    def __init__(self, path):
        self.path = path
        self.parser = configparser.ConfigParser()
        self.read(path)

    def read(self, path):
        self.parser.read(path)

    def save(self):
        with open(self.path, 'w') as configfile:
            self.parser.write(configfile)

    def toDict(self):
        out = {}
        for s in self:
            d = {}
            for o in self[s]:
                d[o] = self.get(s, o)
            out[s] = d
        return out


config = CustomConfig(pathToConfig)

if os.environ.get('DEVMODE') == '1':
    config.read(pathToConfigOverrides)
    config.update()