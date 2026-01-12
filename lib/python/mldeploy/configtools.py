from configparser import ConfigParser, NoOptionError
import inspect
import sys
from types import ModuleType


class Section():
    """
    Dummy class. Subclass this to mark a class as a section. 
    """
    pass


class Option():
    def __init__(self, type, default=None):
        self.type = type
        self.set(default)

    def get(self):
        return self._value

    def set(self, value):
        if isinstance(value, self.type) or isinstance(value, type(None)):
            self._value = value
        else:
            raise TypeError('Assignment failed: Tried to assign %s to %s-type config var (None is also valid)' % (
                type(value).__name__, self.type.__name__))

class ConfigManager():
    def __init__(self, module: ModuleType = None):
        if module == None:
            try:
                frame_infos = inspect.stack()
                for frame_info in frame_infos:
                    module = inspect.getmodule(frame_info.frame)
                    if module and module is not sys.modules[__name__]:
                        self.module = module
                        break
            except Exception:
                raise AttributeError('Module where ConfigManager was instanced could not be found')
        elif isinstance(module, ModuleType):
            self.module = module
        else: raise TypeError('expected ModuleType but got %s' % type(module).__name__)

    def get_sections(self):
        sections = []
        for item in self.module.__dict__.values():
            if inspect.isclass(item):
                if item.__base__ == Section:
                    sections.append(item)
        return sections

    def load(self, file_paths, force_compl=False):
        _parser = ConfigParser()
        _parser.read(file_paths)
        sections = self.get_sections()

        for section in sections:
            section_name = section.__name__
            for option_name, option in list(section.__dict__.items()):
                if isinstance(option, Option):

                    try:
                        if option.type == str:
                            option.set(_parser.get(
                                section_name, option_name))

                        if option.type == int:
                            option.set(_parser.getint(
                                section_name, option_name))

                        if option.type == float:
                            option.set(_parser.getfloat(
                                section_name, option_name))

                        if option.type == bool:
                            option.set(_parser.getboolean(
                                section_name, option_name))
                    except ValueError as err:
                        if not err.args:
                            err.args = ('',)
                        err.args = err.args + ('occurred in %s %s' % (section_name, option_name),)
                        raise
                    except NoOptionError:
                        if force_compl:
                            raise

    def save(self, file_path):
        _parser = ConfigParser()
        sections = self.get_sections()

        for section in sections:
            section_name = section.__name__
            for option_name, option in list(section.__dict__.items()):
                if isinstance(option, Option):

                    if not _parser.has_section(section_name):
                        _parser.add_section(section_name)
                    if option.get():
                        _parser.set(section_name, option_name,
                                    str(option.get()))

        with open(file_path, 'w') as configfile:
            _parser.write(configfile)

