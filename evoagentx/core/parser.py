from .module import BaseModule


class Parser(BaseModule):

    @classmethod
    def parse(cls, content: str, **kwargs):
        """
        the method used to parse text into a Parser object. Use Parser.from_str to parse input by default. 
        """
        return cls.from_str(content, **kwargs)
    
    def save(self, path: str, **kwargs)-> str:
        super().save_module(path, **kwargs)



    
