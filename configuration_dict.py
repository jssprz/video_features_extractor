class ConfigDict:
    def __init__(self, dictionary):
        self.parse_config(dictionary)

    def parse_config(self, dictionary):
        for k, v in dictionary.items():
            if type(v) == dict:
                setattr(self, k, ConfigDict(v))
            else:
                setattr(self, k, v)