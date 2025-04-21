from pydantic import BaseModel, Field


class Document(BaseModel):
    """
    A document is a single piece of text with a title.

    """

    title: str = Field(default=None, description="the title of a document")
    text: str = Field(description="[required] the text of the document.")

    def __str__(self):
        """
        return the string of a document
        """
        return self.to_str()
    
    def to_str(self):
        """
        return the string of a document
        """
        if self.title is None:
            result = self.text 
        else:
            result = "title: " + self.title + ". text: " + self.text
        return result


class Corpus:
    pass

