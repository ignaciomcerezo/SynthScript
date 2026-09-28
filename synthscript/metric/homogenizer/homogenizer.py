from abc import ABC, abstractmethod


class TextHomogenizer(ABC):

    @abstractmethod
    def homogenize(self, text: str) -> str:
        raise NotImplementedError

    def __call__(self, text: str) -> str:
        """
        Homogenizes a string using self.homogenize() (alias).
        """
        return self.homogenize(text)
