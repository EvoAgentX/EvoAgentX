import unittest
from pydantic import Field
from evoagentx.models.base_model import LLMOutputParser


class TestParserWithoutContent(LLMOutputParser):
    """Test parser without user-defined content field."""
    result: str = Field(description="test result")
    score: int = Field(description="test score")


class TestParserWithContent(LLMOutputParser):
    """Test parser with user-defined content field."""
    content: str = Field(description="user defined content field")
    result: str = Field(description="test result")
    score: int = Field(description="test score")

class TestParserWithContent2(TestParserWithContent):
    """Test parser with user-defined content field."""
    confidence: int = Field(description="confidence score")


class TestContentFieldConflict(unittest.TestCase):
    """Test cases for content field conflict resolution."""

    def test_helper_method_detection(self):
        """Test the _is_content_defined_in_subclass helper method."""
        # Should return False for parser without user-defined content
        self.assertFalse(TestParserWithoutContent._is_content_defined_in_subclass())
        
        # Should return True for parser with user-defined content
        self.assertTrue(TestParserWithContent._is_content_defined_in_subclass())

        # Should return True for parser with user-defined content in subclass
        self.assertTrue(TestParserWithContent2._is_content_defined_in_subclass())
        
        # Should return False for base class
        self.assertFalse(LLMOutputParser._is_content_defined_in_subclass())

    def test_get_attrs_without_user_content(self):
        """Test get_attrs for parser without user-defined content field."""
        attrs = TestParserWithoutContent.get_attrs()
        
        # Should contain user-defined fields
        self.assertIn("result", attrs)
        self.assertIn("score", attrs)
        
        # Should NOT contain 'content' (excluded by base class)
        self.assertNotIn("content", attrs)
        
        # Should NOT contain 'class_name' (always excluded)
        self.assertNotIn("class_name", attrs)

    def test_get_attrs_with_user_content(self):
        """Test get_attrs for parser with user-defined content field."""
        attrs = TestParserWithContent.get_attrs()
        
        # Should contain all user-defined fields including content
        self.assertIn("content", attrs)
        self.assertIn("result", attrs)
        self.assertIn("score", attrs)
        
        # Should NOT contain 'class_name' (always excluded)
        self.assertNotIn("class_name", attrs)

    def test_parse_without_user_content(self):
        """Test parsing when no user-defined content field exists."""
        test_json = '{"result": "success", "score": 95}'
        
        parser = TestParserWithoutContent.parse(test_json, parse_mode="json")
        
        # Should parse user fields correctly
        self.assertEqual(parser.result, "success")
        self.assertEqual(parser.score, 95)
        
        # Should store raw LLM output in base class content field
        self.assertEqual(parser.content, test_json)
        self.assertEqual(parser.to_str(), test_json)

    def test_parse_with_user_content(self):
        """Test parsing when user-defined content field exists."""
        test_json = '{"content": "user content", "result": "success", "score": 95}'
        
        parser = TestParserWithContent.parse(test_json, parse_mode="json")
        
        # Should parse all user fields correctly
        self.assertEqual(parser.content, "user content")  # User's content field
        self.assertEqual(parser.result, "success")
        self.assertEqual(parser.score, 95)
        
        # Raw LLM output should still be accessible via to_str()
        self.assertEqual(parser.to_str(), test_json)

    def test_get_attrs_with_return_type(self):
        """Test get_attrs with return_type=True."""
        # Without user content
        attrs_without = TestParserWithoutContent.get_attrs(return_type=True)
        attr_names_without = [attr[0] for attr in attrs_without]
        
        self.assertIn("result", attr_names_without)
        self.assertIn("score", attr_names_without)
        self.assertNotIn("content", attr_names_without)
        
        # With user content
        attrs_with = TestParserWithContent.get_attrs(return_type=True)
        attr_names_with = [attr[0] for attr in attrs_with]
        
        self.assertIn("content", attr_names_with)
        self.assertIn("result", attr_names_with)
        self.assertIn("score", attr_names_with)

    def test_get_attr_descriptions(self):
        """Test get_attr_descriptions method."""
        # Without user content
        descriptions_without = TestParserWithoutContent.get_attr_descriptions()
        self.assertIn("result", descriptions_without)
        self.assertIn("score", descriptions_without)
        self.assertNotIn("content", descriptions_without)
        
        # With user content
        descriptions_with = TestParserWithContent.get_attr_descriptions()
        self.assertIn("content", descriptions_with)
        self.assertIn("result", descriptions_with)
        self.assertIn("score", descriptions_with)
        self.assertEqual(descriptions_with["content"], "user defined content field")

    def test_get_specification(self):
        """Test get_specification method."""
        # Without user content
        spec_without = TestParserWithoutContent.get_specification()
        self.assertIsInstance(spec_without, str)
        self.assertIn("result", spec_without)
        self.assertIn("score", spec_without)
        self.assertNotIn("content", spec_without)
        
        # With user content
        spec_with = TestParserWithContent.get_specification()
        self.assertIsInstance(spec_with, str)
        self.assertIn("content", spec_with)
        self.assertIn("result", spec_with)
        self.assertIn("score", spec_with)

    def test_get_structured_data(self):
        """Test get_structured_data method."""
        # Without user content
        test_json1 = '{"result": "success", "score": 95}'
        parser1 = TestParserWithoutContent.parse(test_json1, parse_mode="json")
        structured_data1 = parser1.get_structured_data()
        
        self.assertIn("result", structured_data1)
        self.assertIn("score", structured_data1)
        self.assertNotIn("content", structured_data1)
        self.assertEqual(structured_data1["result"], "success")
        self.assertEqual(structured_data1["score"], 95)
        
        # With user content
        test_json2 = '{"content": "user content", "result": "success", "score": 95}'
        parser2 = TestParserWithContent.parse(test_json2, parse_mode="json")
        structured_data2 = parser2.get_structured_data()
        
        self.assertIn("content", structured_data2)
        self.assertIn("result", structured_data2)
        self.assertIn("score", structured_data2)
        self.assertEqual(structured_data2["content"], "user content")
        self.assertEqual(structured_data2["result"], "success")
        self.assertEqual(structured_data2["score"], 95)



if __name__ == "__main__":
    unittest.main()
