from abc import ABCMeta, abstractmethod


class AbsSampler(metaclass=ABCMeta):
    @abstractmethod
    def sampling(self, *args, **kwargs):
        pass

    @abstractmethod
    def data(self):
        pass
