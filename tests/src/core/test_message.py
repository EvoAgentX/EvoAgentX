import unittest
import time
from evoagentx.core.message import Message, MessageType

class ToyContent:

    def __init__(self, content: str):
        self.content = content 
    
    def __str__(self) -> str:
        return self.to_str()
    
    def to_str(self) -> str:
        return self.content
    

class TestModule(unittest.TestCase):

    def test_message(self):

        m1 = Message(content = "test_content", agent = "agent1", action = "action1", next_actions = ["action2"], msg_type=MessageType.REQUEST)
        time.sleep(5)
        m2 = Message(content=ToyContent(content="test_content2"), agent="agent2", action="action3", msg_type=MessageType.RESPONSE)
        time.sleep(5)
        m3 = Message(content = "test_content", agent = "agent1", action = "action1", next_actions = ["action2"], msg_type=MessageType.REQUEST)

        self.assertTrue(m3 != m1)
        m3_message_id = m3.message_id
        m3.message_id = m1.message_id
        self.assertTrue(m1 == m3)
        m3.message_id = m3_message_id

        message_str = str(m2)
        self.assertTrue("Content: test_content2" in message_str)

        sorted_message_based_on_timestamp = Message.sort([m3, m2, m1])
        self.assertEqual(sorted_message_based_on_timestamp[0].message_id, m1.message_id)
        self.assertEqual(sorted_message_based_on_timestamp[1].message_id, m2.message_id)
        self.assertEqual(sorted_message_based_on_timestamp[2].message_id, m3.message_id)

        merged_message = Message.merge([[m3], [m1, m2]], sort=True)
        self.assertEqual(merged_message[0].message_id, m1.message_id)
        self.assertEqual(merged_message[1].message_id, m2.message_id)
        self.assertEqual(merged_message[2].message_id, m3.message_id)
        

if __name__ == "__main__":
    unittest.main()