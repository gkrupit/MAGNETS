# ExperimentData

class ExperimentData:
    def __init__(self, attributes, values):
        self.attributes = dict()
        for a,v in zip(attributes, values):
            self.attributes[a] = v

    def Get(self, attribute):
        return self.attributes[attribute]

    def Set(self, attribute, value):
        self.attributes[attribute] = value

    def HasValues(self, attributes):
        return all(self.Get(a) for a in attributes )

    def GetMultiple(self, attributes):
        if not self.HasValues(attributes):
            return None

        to_return = []
        for a in attributes:
            to_return.append(self.Get(a))
        return to_return

    @staticmethod
    def GetMultipleFrom(experiment_data, attributes):
        return [ e.GetMultiple(attributes) for e in experiment_data ]

    @staticmethod
    def GetFrom(experiment_data, attribute):
        return [ e.Get(attribute) for e in experiment_data ]


        return [ self.Get(a) for a in attributes ]
