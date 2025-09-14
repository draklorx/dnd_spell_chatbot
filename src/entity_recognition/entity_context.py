class EntityContext:
    def __init__(self):
        self.context = {}

    def update(self, name, value):
        self.context[name] = value

    def get(self, name):
        return self.context.get(name, None)