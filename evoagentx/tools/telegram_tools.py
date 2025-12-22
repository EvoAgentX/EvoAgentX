"""
Telegram API Tool for EvoAgentX

This module provides comprehensive Telegram integration including:
- Message retrieval and search
- Sending messages and scheduling
- Chat management and file operations

Compatible with EvoAgentX tool architecture and follows the latest Telegram API patterns.
"""

import os
import asyncio
import time
from typing import Dict, Any, List
from telethon import TelegramClient
from telethon.tl.types import Message, User, Chat, Channel
from telethon.errors import (
    FloodWaitError,
    ChatAdminRequiredError,
    UserBannedInChannelError,
    ChannelPrivateError,
    UserNotParticipantError,
    ChatWriteForbiddenError,
    MessageEmptyError,
    MessageTooLongError
)
from dotenv import load_dotenv
import PyPDF2

from .tool import Tool, Toolkit
from ..core.module import BaseModule
from ..core.logging import logger

# Load environment variables
load_dotenv()

# Global constants
SESSION_NAME = 'ai_agent_session'


class TelegramBase(BaseModule):
    """
    Base class for Telegram API interactions.
    Handles client management, authentication, and common utilities.
    """
    
    def __init__(self, api_id: str = None, api_hash: str = None, phone: str = None, **kwargs):
        """
        Initialize the Telegram base.
        
        Args:
            api_id (str, optional): Telegram API ID. If not provided, will try to get from TELEGRAM_API_ID environment variable.
            api_hash (str, optional): Telegram API Hash. If not provided, will try to get from TELEGRAM_API_HASH environment variable.
            phone (str, optional): Phone number for authentication. If not provided, will try to get from TELEGRAM_PHONE environment variable.
            **kwargs: Additional keyword arguments for parent class
        """
        super().__init__(**kwargs)
        
        # Get credentials from parameters or environment variables
        self.api_id = api_id or os.getenv("TELEGRAM_API_ID")
        self.api_hash = api_hash or os.getenv("TELEGRAM_API_HASH")
        self.phone = phone or os.getenv("TELEGRAM_PHONE")
        
        if not self.api_id or not self.api_hash:
            logger.warning(
                "No Telegram API credentials provided. Please set TELEGRAM_API_ID and TELEGRAM_API_HASH environment variables "
                "or pass api_id and api_hash parameters. Get your credentials from: https://my.telegram.org/apps"
            )
    
    def _get_client(self) -> TelegramClient:
        """
        Create and return a Telegram client instance.
        
        Returns:
            TelegramClient: Configured Telegram client
        """
        if not self.api_id or not self.api_hash:
            raise ValueError("Telegram API credentials not found. Please set TELEGRAM_API_ID and TELEGRAM_API_HASH environment variables.")
        
        client = TelegramClient(SESSION_NAME, self.api_id, self.api_hash)
        return client
    
    def _format_message(self, message: Message) -> Dict[str, Any]:
        """
        Format a Telegram message for consistent output.
        
        Args:
            message: Telegram message object
            
        Returns:
            dict: Formatted message data
        """
        return {
            "id": message.id,
            "text": message.text or "",
            "date": message.date.isoformat() if message.date else None,
            "sender_id": message.sender_id,
            "chat_id": message.chat_id,
            "is_reply": message.reply_to_msg_id is not None,
            "reply_to_msg_id": message.reply_to_msg_id,
            "has_media": message.media is not None,
            "media_type": type(message.media).__name__ if message.media else None
        }
    
    def _format_chat(self, chat) -> Dict[str, Any]:
        """
        Format a Telegram chat for consistent output.
        
        Args:
            chat: Telegram chat object
            
        Returns:
            dict: Formatted chat data
        """
        chat_type = "unknown"
        title = "Unknown"
        
        if isinstance(chat, User):
            chat_type = "user"
            title = f"{chat.first_name or ''} {chat.last_name or ''}".strip() or chat.username or "Unknown User"
        elif isinstance(chat, Chat):
            chat_type = "group"
            title = chat.title or "Unknown Group"
        elif isinstance(chat, Channel):
            chat_type = "channel" if chat.broadcast else "supergroup"
            title = chat.title or "Unknown Channel"
        
        return {
            "id": chat.id,
            "title": title,
            "type": chat_type,
            "username": getattr(chat, 'username', None)
        }
    
    def _run_async(self, coro):
        """
        Run an async coroutine, handling both sync and async contexts.
        
        Args:
            coro: Async coroutine to run
            
        Returns:
            Result of the coroutine
        """
        try:
            try:
                asyncio.get_running_loop()
                # We're in an async context, need to run in a new thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, coro)
                    return future.result()
            except RuntimeError:
                # No running loop, we can use asyncio.run
                return asyncio.run(coro)
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to execute async operation: {str(e)}"
            }


class FetchLatestMessagesTool(Tool):
    """Retrieve the most recent messages from a specific Telegram contact by their name."""
    
    name: str = "fetch_latest_messages"
    description: str = "Retrieve the most recent messages from a specific Telegram contact by their name. If multiple contacts match the name, it will ask for clarification."
    inputs: Dict[str, Dict[str, str]] = {
        "contact_name": {
            "type": "string",
            "description": "The name of the contact to fetch messages from (e.g., 'Shivam Kumar')"
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of messages to retrieve (default: 10)"
        }
    }
    required: List[str] = ["contact_name"]
    
    def __init__(self, telegram_base: TelegramBase):
        super().__init__()
        self.telegram_base = telegram_base
    
    def __call__(self, contact_name: str, limit: int = 10) -> Dict[str, Any]:
        """
        Fetch the latest messages from a Telegram contact by their name.
        
        Args:
            contact_name: The name of the Telegram contact
            limit: Maximum number of messages to retrieve
            
        Returns:
            Dictionary with message results
        """
        async def _fetch_messages():
            client = None
            try:
                client = self.telegram_base._get_client()
                await client.start(phone=self.telegram_base.phone)
                
                # Find contact by name (users, groups, and channels)
                matches = []
                async for dialog in client.iter_dialogs():
                    if contact_name.lower() in dialog.name.lower():
                        matches.append({"name": dialog.name, "id": dialog.id, "chat": dialog.entity})
                
                if len(matches) == 0:
                    return {
                        "success": False,
                        "error": f"Contact '{contact_name}' not found. Please check the name."
                    }
                
                if len(matches) > 1:
                    # Format the list of matches for the user to choose from
                    clarification_list = [f"{m['name']} (ID: {m['id']})" for m in matches]
                    return {
                        "success": False,
                        "error": "Ambiguous contact name. Please clarify which user you mean.",
                        "clarification_needed": clarification_list
                    }
                
                # If we reach here, we have exactly one match
                chat = matches[0]['chat']
                
                # Fetch messages
                messages = []
                async for message in client.iter_messages(chat, limit=limit):
                    messages.append(self.telegram_base._format_message(message))
                
                # Format chat info
                chat_info = self.telegram_base._format_chat(chat)
                
                return {
                    "success": True,
                    "chat": chat_info,
                    "messages_count": len(messages),
                    "messages": messages
                }
                    
            except FloodWaitError as e:
                return {
                    "success": False,
                    "error": f"Rate limited. Please wait {e.seconds} seconds before trying again."
                }
            except (ChatAdminRequiredError, UserBannedInChannelError, ChannelPrivateError, 
                   UserNotParticipantError, ChatWriteForbiddenError) as e:
                return {
                    "success": False,
                    "error": f"Access denied: {str(e)}"
                }
            except Exception as e:
                logger.error(f"Error fetching messages: {str(e)}")
                return {
                    "success": False,
                    "error": f"Failed to fetch messages: {str(e)}"
                }
            finally:
                if client:
                    await client.disconnect()
        
        # Run the async function
        return self.telegram_base._run_async(_fetch_messages())


class SearchMessagesByKeywordTool(Tool):
    """Find specific information by searching for a keyword within a contact's chat history."""
    
    name: str = "search_messages_by_keyword"
    description: str = "Find specific information by searching for a keyword within a contact's chat history. If multiple contacts match the name, it will ask for clarification."
    inputs: Dict[str, Dict[str, str]] = {
        "contact_name": {
            "type": "string",
            "description": "The name of the contact to search messages from (e.g., 'Shivam Kumar')"
        },
        "keyword": {
            "type": "string",
            "description": "Keyword or phrase to search for in messages"
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of matching messages to retrieve (default: 10)"
        }
    }
    required: List[str] = ["contact_name", "keyword"]
    
    def __init__(self, telegram_base: TelegramBase):
        super().__init__()
        self.telegram_base = telegram_base
    
    def __call__(self, contact_name: str, keyword: str, limit: int = 10) -> Dict[str, Any]:
        """
        Search for messages containing a specific keyword in a contact's chat.
        
        Args:
            contact_name: The name of the Telegram contact
            keyword: Keyword to search for
            limit: Maximum number of matching messages to retrieve
            
        Returns:
            Dictionary with search results
        """
        async def _search_messages():
            client = None
            try:
                client = self.telegram_base._get_client()
                await client.start(phone=self.telegram_base.phone)
                
                # Find contact by name (users, groups, and channels)
                matches = []
                async for dialog in client.iter_dialogs():
                    if contact_name.lower() in dialog.name.lower():
                        matches.append({"name": dialog.name, "id": dialog.id, "chat": dialog.entity})
                
                if len(matches) == 0:
                    return {
                        "success": False,
                        "error": f"Contact '{contact_name}' not found. Please check the name."
                    }
                
                if len(matches) > 1:
                    # Format the list of matches for the user to choose from
                    clarification_list = [f"{m['name']} (ID: {m['id']})" for m in matches]
                    return {
                        "success": False,
                        "error": "Ambiguous contact name. Please clarify which user you mean.",
                        "clarification_needed": clarification_list
                    }
                
                # If we reach here, we have exactly one match
                chat = matches[0]['chat']
                
                # Search messages
                messages = []
                async for message in client.iter_messages(chat, search=keyword, limit=limit):
                    if message.text and keyword.lower() in message.text.lower():
                        messages.append(self.telegram_base._format_message(message))
                
                # Format chat info
                chat_info = self.telegram_base._format_chat(chat)
                
                return {
                    "success": True,
                    "chat": chat_info,
                    "keyword": keyword,
                    "matches_count": len(messages),
                    "messages": messages
                }
                    
            except FloodWaitError as e:
                return {
                    "success": False,
                    "error": f"Rate limited. Please wait {e.seconds} seconds before trying again."
                }
            except (ChatAdminRequiredError, UserBannedInChannelError, ChannelPrivateError, 
                   UserNotParticipantError, ChatWriteForbiddenError) as e:
                return {
                    "success": False,
                    "error": f"Access denied: {str(e)}"
                }
            except Exception as e:
                logger.error(f"Error searching messages: {str(e)}")
                return {
                    "success": False,
                    "error": f"Failed to search messages: {str(e)}"
                }
            finally:
                if client:
                    await client.disconnect()
        
        # Run the async function
        return self.telegram_base._run_async(_search_messages())


class SendMessageTool(Tool):
    """Send a text message to a Telegram contact by their name."""
    
    # --- MODIFIED TOOL DEFINITION ---
    name: str = "send_message_by_name"
    description: str = (
        "Finds a contact by their name and sends them a text message. "
        "If multiple contacts match the name, it will ask for clarification."
    )
    inputs: Dict[str, Dict[str, str]] = {
        "contact_name": {
            "type": "string",
            "description": "The name of the contact to search for (e.g., 'Shivam Kumar')"
        },
        "message_text": {
            "type": "string",
            "description": "The text message to send"
        }
    }
    required: List[str] = ["contact_name", "message_text"]
    
    def __init__(self, telegram_base: TelegramBase):
        super().__init__()
        self.telegram_base = telegram_base
    
    # --- MODIFIED CALL SIGNATURE ---
    def __call__(self, contact_name: str, message_text: str) -> Dict[str, Any]:
        """
        Finds a contact by name and sends them a message.
        
        Args:
            contact_name: The name of the Telegram contact.
            message_text: Text message to send.
            
        Returns:
            Dictionary with the send result or a request for clarification.
        """
        async def _send_message_by_name():
            client = None
            try:
                client = self.telegram_base._get_client()
                await client.start(phone=self.telegram_base.phone)
                
                # --- NEW LOGIC START: Find Contact by Name ---
                matches = []
                async for dialog in client.iter_dialogs():
                    if dialog.is_user and not dialog.entity.bot:
                        if contact_name.lower() in dialog.name.lower():
                            matches.append({"name": dialog.name, "id": dialog.id})
                
                if len(matches) == 0:
                    return {
                        "success": False,
                        "error": f"Contact '{contact_name}' not found. Please check the name."
                    }
                
                if len(matches) > 1:
                    # Format the list of matches for the user to choose from
                    clarification_list = [f"{m['name']} (ID: {m['id']})" for m in matches]
                    return {
                        "success": False,
                        "error": "Ambiguous contact name. Please clarify which user you mean.",
                        "clarification_needed": clarification_list
                    }
                
                # If we reach here, we have exactly one match.
                chat_id = matches[0]['id']
                # --- NEW LOGIC END ---
                
                # Send message using the resolved chat_id
                sent_message = await client.send_message(chat_id, message_text)
                
                # Get entity for formatting the response
                chat = await client.get_entity(chat_id)
                chat_info = self.telegram_base._format_chat(chat)
                
                return {
                    "success": True,
                    "message_id": sent_message.id,
                    "chat": chat_info,
                    "message_text": message_text,
                    "sent_at": sent_message.date.isoformat() if sent_message.date else None
                }
            
            # --- REUSED EXISTING ERROR HANDLING ---
            except FloodWaitError as e:
                return {"success": False, "error": f"Rate limited. Please wait {e.seconds} seconds."}
            except (ChatAdminRequiredError, UserBannedInChannelError, ChannelPrivateError, 
                    UserNotParticipantError, ChatWriteForbiddenError) as e:
                return {"success": False, "error": f"Access denied: {str(e)}"}
            except MessageEmptyError:
                return {"success": False, "error": "Message is empty."}
            except MessageTooLongError:
                return {"success": False, "error": "Message is too long."}
            except Exception as e:
                logger.error(f"Error sending message: {str(e)}")
                return {"success": False, "error": f"Failed to send message: {str(e)}"}
            finally:
                if client:
                    await client.disconnect()
        
        return self.telegram_base._run_async(_send_message_by_name())


class ListRecentChatsTool(Tool):
    """Get a list of recent conversations, allowing the agent to ask for clarification if a user's request is ambiguous."""
    
    name: str = "list_recent_chats"
    description: str = "Get a list of recent conversations, allowing the agent to ask for clarification if a user's request is ambiguous (e.g., 'Summarize my last chat')."
    inputs: Dict[str, Dict[str, str]] = {
        "limit": {
            "type": "integer",
            "description": "Maximum number of recent chats to retrieve (default: 10)"
        }
    }
    required: List[str] = []
    
    def __init__(self, telegram_base: TelegramBase):
        super().__init__()
        self.telegram_base = telegram_base
    
    def __call__(self, limit: int = 10) -> Dict[str, Any]:
        """
        List recent Telegram chats.
        
        Args:
            limit: Maximum number of recent chats to retrieve
            
        Returns:
            Dictionary with chat list
        """
        async def _list_chats():
            client = None
            try:
                client = self.telegram_base._get_client()
                await client.start(phone=self.telegram_base.phone)
                
                # Get recent dialogs
                dialogs = []
                async for dialog in client.iter_dialogs(limit=limit):
                    chat_info = self.telegram_base._format_chat(dialog.entity)
                    dialogs.append({
                        **chat_info,
                        "last_message_date": dialog.date.isoformat() if dialog.date else None,
                        "unread_count": dialog.unread_count
                    })
                
                return {
                    "success": True,
                    "chats_count": len(dialogs),
                    "chats": dialogs
                }
                    
            except Exception as e:
                logger.error(f"Error listing chats: {str(e)}")
                return {
                    "success": False,
                    "error": f"Failed to list chats: {str(e)}"
                }
            finally:
                if client:
                    await client.disconnect()
        
        # Run the async function
        return self.telegram_base._run_async(_list_chats())


class FindAndRetrieveFileTool(Tool):
    """Locate a specific file within a contact's chat based on a search query. This tool should return metadata about the file (name, size, type), not download its contents."""
    
    name: str = "find_and_retrieve_file"
    description: str = "Locate a specific file within a contact's chat based on a search query. This tool should return metadata about the file (name, size, type), not download its contents. If multiple contacts match the name, it will ask for clarification."
    inputs: Dict[str, Dict[str, str]] = {
        "contact_name": {
            "type": "string",
            "description": "The name of the contact to search files from (e.g., 'Shivam Kumar')"
        },
        "filename_query": {
            "type": "string",
            "description": "Filename or search query to find files"
        }
    }
    required: List[str] = ["contact_name", "filename_query"]
    
    def __init__(self, telegram_base: TelegramBase):
        super().__init__()
        self.telegram_base = telegram_base
    
    def __call__(self, contact_name: str, filename_query: str) -> Dict[str, Any]:
        """
        Find files in a Telegram contact's chat based on filename query.
        
        Args:
            contact_name: The name of the Telegram contact
            filename_query: Filename or search query to find files
            
        Returns:
            Dictionary with file search results
        """
        async def _find_files():
            client = None
            try:
                client = self.telegram_base._get_client()
                await client.start(phone=self.telegram_base.phone)
                
                # Find contact by name (users, groups, and channels)
                matches = []
                async for dialog in client.iter_dialogs():
                    if contact_name.lower() in dialog.name.lower():
                        matches.append({"name": dialog.name, "id": dialog.id, "chat": dialog.entity})
                
                if len(matches) == 0:
                    return {
                        "success": False,
                        "error": f"Contact '{contact_name}' not found. Please check the name."
                    }
                
                if len(matches) > 1:
                    # Format the list of matches for the user to choose from
                    clarification_list = [f"{m['name']} (ID: {m['id']})" for m in matches]
                    return {
                        "success": False,
                        "error": "Ambiguous contact name. Please clarify which user you mean.",
                        "clarification_needed": clarification_list
                    }
                
                # If we reach here, we have exactly one match
                chat = matches[0]['chat']
                
                # Search for files using the working Telethon approach
                files = []
                message_count = 0
                
                async for message in client.iter_messages(chat):
                    message_count += 1
                    
                    if message.document:
                        # Get file information using the working approach
                        doc = message.document
                        filename = "Unknown"
                        
                        # Extract filename from attributes (working approach)
                        for attribute in doc.attributes:
                            if hasattr(attribute, 'file_name'):
                                filename = attribute.file_name
                                break
                        
                        # Check if filename matches query (case-insensitive)
                        if not filename_query or filename_query.lower() in filename.lower():
                            files.append({
                                "message_id": message.id,
                                "filename": filename,
                                "file_size": doc.size,
                                "mime_type": doc.mime_type,
                                "date": message.date.isoformat() if message.date else None,
                                "sender_id": message.sender_id,
                                "caption": message.text or ""
                            })
                    
                    # Limit search to prevent infinite loops (increased from 100 to 1000)
                    if message_count > 1000:
                        break
                
                # Format chat info
                chat_info = self.telegram_base._format_chat(chat)
                
                return {
                    "success": True,
                    "chat": chat_info,
                    "query": filename_query,
                    "files_found": len(files),
                    "files": files
                }
                    
            except FloodWaitError as e:
                return {
                    "success": False,
                    "error": f"Rate limited. Please wait {e.seconds} seconds before trying again."
                }
            except (ChatAdminRequiredError, UserBannedInChannelError, ChannelPrivateError, 
                   UserNotParticipantError, ChatWriteForbiddenError) as e:
                return {
                    "success": False,
                    "error": f"Access denied: {str(e)}"
                }
            except Exception as e:
                logger.error(f"Error finding files: {str(e)}")
                return {
                    "success": False,
                    "error": f"Failed to find files: {str(e)}"
                }
            finally:
                if client:
                    await client.disconnect()
        
        # Run the async function
        return self.telegram_base._run_async(_find_files())


class SummarizeContactMessagesTool(Tool):
    """Summarize recent messages from a specific Telegram contact by their name."""
    
    name: str = "summarize_contact_messages"
    description: str = "Summarize recent messages from a specific Telegram contact by their name. Provides a summary of the conversation history."
    inputs: Dict[str, Dict[str, str]] = {
        "contact_name": {
            "type": "string",
            "description": "The name of the contact to summarize messages for (e.g., 'Shivam Kumar')"
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of recent messages to analyze for summarization (default: 20)"
        }
    }
    required: List[str] = ["contact_name"]
    
    def __init__(self, telegram_base: TelegramBase):
        super().__init__()
        self.telegram_base = telegram_base
    
    def __call__(self, contact_name: str, limit: int = 20) -> Dict[str, Any]:
        """
        Summarize recent messages from a contact by name.
        
        Args:
            contact_name: The name of the Telegram contact
            limit: Maximum number of recent messages to analyze
            
        Returns:
            Dictionary with summarization results
        """
        async def _summarize_messages():
            client = None
            try:
                client = self.telegram_base._get_client()
                await client.start(phone=self.telegram_base.phone)
                
                # Find contact by name (same logic as send_message_by_name)
                matches = []
                async for dialog in client.iter_dialogs():
                    if dialog.is_user and not dialog.entity.bot:
                        if contact_name.lower() in dialog.name.lower():
                            matches.append({"name": dialog.name, "id": dialog.id})
                
                if len(matches) == 0:
                    return {
                        "success": False,
                        "error": f"Contact '{contact_name}' not found. Please check the name."
                    }
                
                if len(matches) > 1:
                    # Format the list of matches for the user to choose from
                    clarification_list = [f"{m['name']} (ID: {m['id']})" for m in matches]
                    return {
                        "success": False,
                        "error": "Ambiguous contact name. Please clarify which user you mean.",
                        "clarification_needed": clarification_list
                    }
                
                # If we reach here, we have exactly one match.
                chat_id = matches[0]['id']
                
                # Get chat entity for formatting
                chat = await client.get_entity(chat_id)
                chat_info = self.telegram_base._format_chat(chat)
                
                # Fetch recent messages
                messages = []
                async for message in client.iter_messages(chat, limit=limit):
                    if message.text:  # Only include text messages
                        messages.append({
                            "id": message.id,
                            "text": message.text,
                            "date": message.date.isoformat() if message.date else None,
                            "sender_id": message.sender_id,
                            "is_outgoing": message.out
                        })
                
                # Create a simple summary
                if not messages:
                    summary = f"No recent text messages found with {contact_name}."
                else:
                    # Basic summarization logic
                    total_messages = len(messages)
                    outgoing_count = sum(1 for msg in messages if msg['is_outgoing'])
                    incoming_count = total_messages - outgoing_count
                    
                    # Get date range
                    dates = [msg['date'] for msg in messages if msg['date']]
                    if dates:
                        latest_date = max(dates)
                        earliest_date = min(dates)
                    else:
                        latest_date = earliest_date = "Unknown"
                    
                    # Extract key topics (simple keyword extraction)
                    all_text = " ".join([msg['text'] for msg in messages])
                    words = all_text.lower().split()
                    word_freq = {}
                    for word in words:
                        if len(word) > 3:  # Only words longer than 3 characters
                            word_freq[word] = word_freq.get(word, 0) + 1
                    
                    # Get top 5 most frequent words
                    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
                    
                    summary = f"""Conversation Summary with {contact_name}:
• Total messages analyzed: {total_messages}
• Messages from you: {outgoing_count}
• Messages from {contact_name}: {incoming_count}
• Date range: {earliest_date} to {latest_date}
• Key topics: {', '.join([word for word, freq in top_words])}
• Recent activity: {'Active' if total_messages > 0 else 'No recent messages'}"""
                
                return {
                    "success": True,
                    "contact": chat_info,
                    "messages_analyzed": len(messages),
                    "summary": summary,
                    "recent_messages": messages[:5]  # Show first 5 messages as examples
                }
                    
            except Exception as e:
                logger.error(f"Error summarizing messages: {str(e)}")
                return {
                    "success": False,
                    "error": f"Failed to summarize messages: {str(e)}"
                }
            finally:
                if client:
                    await client.disconnect()
        
        # Run the async function
        return self.telegram_base._run_async(_summarize_messages())


class DownloadFileTool(Tool):
    """Download a file from a Telegram contact by their name."""
    
    name: str = "download_file"
    description: str = "Download a file from a Telegram contact by their name. Downloads the file to a local directory."
    inputs: Dict[str, Dict[str, str]] = {
        "contact_name": {
            "type": "string",
            "description": "The name of the contact to download file from (e.g., 'Vinay Kumar')"
        },
        "filename_query": {
            "type": "string",
            "description": "Filename or search query to find the file (e.g., 'Kafka.pdf')"
        },
        "download_dir": {
            "type": "string",
            "description": "Directory to download the file to (default: 'downloads')"
        }
    }
    required: List[str] = ["contact_name", "filename_query"]
    
    def __init__(self, telegram_base: TelegramBase):
        super().__init__()
        self.telegram_base = telegram_base
    
    def __call__(self, contact_name: str, filename_query: str, download_dir: str = "downloads") -> Dict[str, Any]:
        """
        Download a file from a Telegram contact.
        
        Args:
            contact_name: The name of the Telegram contact
            filename_query: Filename or search query to find the file
            download_dir: Directory to download the file to
            
        Returns:
            Dictionary with download result
        """
        async def _download_file():
            client = None
            try:
                client = self.telegram_base._get_client()
                await client.start(phone=self.telegram_base.phone)
                
                # Find contact by name
                matches = []
                async for dialog in client.iter_dialogs():
                    if contact_name.lower() in dialog.name.lower():
                        matches.append({"name": dialog.name, "id": dialog.id, "chat": dialog.entity})
                
                if len(matches) == 0:
                    return {
                        "success": False,
                        "error": f"Contact '{contact_name}' not found. Please check the name."
                    }
                
                if len(matches) > 1:
                    clarification_list = [f"{m['name']} (ID: {m['id']})" for m in matches]
                    return {
                        "success": False,
                        "error": "Ambiguous contact name. Please clarify which user you mean.",
                        "clarification_needed": clarification_list
                    }
                
                # Get the contact
                chat = matches[0]['chat']
                
                # Search for the file
                found_message = None
                message_count = 0
                
                async for message in client.iter_messages(chat):
                    message_count += 1
                    
                    if message.document:
                        doc = message.document
                        filename = "Unknown"
                        
                        # Extract filename from attributes
                        for attribute in doc.attributes:
                            if hasattr(attribute, 'file_name'):
                                filename = attribute.file_name
                                break
                        
                        # Check if filename matches query
                        if filename_query.lower() in filename.lower():
                            found_message = message
                            break
                    
                    if message_count > 1000:
                        break
                
                if not found_message:
                    return {
                        "success": False,
                        "error": f"File '{filename_query}' not found in contact '{contact_name}'"
                    }
                
                # Download the file
                if not os.path.exists(download_dir):
                    os.makedirs(download_dir)
                
                downloaded_path = await client.download_media(
                    found_message,
                    file=os.path.join(download_dir, filename)
                )
                
                if downloaded_path:
                    file_size = os.path.getsize(downloaded_path)
                    return {
                        "success": True,
                        "message": "File downloaded successfully",
                        "filename": filename,
                        "file_path": downloaded_path,
                        "file_size": file_size,
                        "download_dir": download_dir,
                        "contact_name": contact_name
                    }
                else:
                    return {
                        "success": False,
                        "error": "File download failed"
                    }
                    
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to download file: {str(e)}"
                }
            finally:
                if client:
                    await client.disconnect()
        
        return self.telegram_base._run_async(_download_file())


class ReadFileContentTool(Tool):
    """Read the content of a file from a Telegram contact by their name."""
    
    name: str = "read_file_content"
    description: str = "Read the content of a file from a Telegram contact by their name. Downloads the file and extracts its text content."
    inputs: Dict[str, Dict[str, str]] = {
        "contact_name": {
            "type": "string",
            "description": "The name of the contact to read file from (e.g., 'Vinay Kumar')"
        },
        "filename_query": {
            "type": "string",
            "description": "Filename or search query to find the file (e.g., 'Kafka.pdf')"
        },
        "content_type": {
            "type": "string",
            "description": "Type of content to extract: 'full', 'first_lines', 'last_lines', 'summary' (default: 'full')"
        },
        "lines_count": {
            "type": "integer",
            "description": "Number of lines to extract for first_lines/last_lines (default: 3)"
        }
    }
    required: List[str] = ["contact_name", "filename_query"]
    
    def __init__(self, telegram_base: TelegramBase):
        super().__init__()
        self.telegram_base = telegram_base
    
    def __call__(self, contact_name: str, filename_query: str, content_type: str = "full", lines_count: int = 3) -> Dict[str, Any]:
        """
        Read the content of a file from a Telegram contact.
        
        Args:
            contact_name: The name of the Telegram contact
            filename_query: Filename or search query to find the file
            content_type: Type of content to extract
            lines_count: Number of lines for first_lines/last_lines
            
        Returns:
            Dictionary with file content
        """
        async def _read_file_content():
            client = None
            try:
                client = self.telegram_base._get_client()
                await client.start(phone=self.telegram_base.phone)
                
                # Find contact by name
                matches = []
                async for dialog in client.iter_dialogs():
                    if contact_name.lower() in dialog.name.lower():
                        matches.append({"name": dialog.name, "id": dialog.id, "chat": dialog.entity})
                
                if len(matches) == 0:
                    return {
                        "success": False,
                        "error": f"Contact '{contact_name}' not found. Please check the name."
                    }
                
                if len(matches) > 1:
                    clarification_list = [f"{m['name']} (ID: {m['id']})" for m in matches]
                    return {
                        "success": False,
                        "error": "Ambiguous contact name. Please clarify which user you mean.",
                        "clarification_needed": clarification_list
                    }
                
                # Get the contact
                chat = matches[0]['chat']
                
                # Search for the file
                found_message = None
                message_count = 0
                
                async for message in client.iter_messages(chat):
                    message_count += 1
                    
                    if message.document:
                        doc = message.document
                        filename = "Unknown"
                        
                        # Extract filename from attributes
                        for attribute in doc.attributes:
                            if hasattr(attribute, 'file_name'):
                                filename = attribute.file_name
                                break
                        
                        # Check if filename matches query
                        if filename_query.lower() in filename.lower():
                            found_message = message
                            break
                    
                    if message_count > 1000:
                        break
                
                if not found_message:
                    return {
                        "success": False,
                        "error": f"File '{filename_query}' not found in contact '{contact_name}'"
                    }
                
                # Download the file temporarily with unique filename
                temp_dir = "temp_downloads"
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)
                
                # Create unique filename to avoid conflicts
                unique_filename = f"{int(time.time())}_{filename}"
                downloaded_path = await client.download_media(
                    found_message,
                    file=os.path.join(temp_dir, unique_filename)
                )
                
                if not downloaded_path:
                    return {
                        "success": False,
                        "error": "Failed to download file for reading"
                    }
                
                # Read file content based on type
                try:
                    if filename.lower().endswith('.pdf'):
                        # Read PDF content
                        with open(downloaded_path, 'rb') as file:
                            pdf_reader = PyPDF2.PdfReader(file)
                            
                            # Extract text from all pages
                            full_text = ""
                            for page in pdf_reader.pages:
                                full_text += page.extract_text() + "\n"
                            
                            # Process content based on type
                            lines = [line.strip() for line in full_text.split('\n') if line.strip()]
                            
                            if content_type == "full":
                                content = full_text
                            elif content_type == "first_lines":
                                content = "\n".join(lines[:lines_count])
                            elif content_type == "last_lines":
                                content = "\n".join(lines[-lines_count:])
                            elif content_type == "summary":
                                content = f"Document has {len(pdf_reader.pages)} pages, {len(lines)} lines, {len(full_text)} characters"
                            else:
                                content = full_text
                            
                            return {
                                "success": True,
                                "message": "File content read successfully",
                                "filename": filename,
                                "content_type": content_type,
                                "content": content,
                                "file_info": {
                                    "pages": len(pdf_reader.pages),
                                    "lines": len(lines),
                                    "characters": len(full_text)
                                },
                                "contact_name": contact_name
                            }
                    else:
                        # Read text file
                        with open(downloaded_path, 'r', encoding='utf-8') as file:
                            content = file.read()
                        
                        lines = content.split('\n')
                        
                        if content_type == "full":
                            processed_content = content
                        elif content_type == "first_lines":
                            processed_content = "\n".join(lines[:lines_count])
                        elif content_type == "last_lines":
                            processed_content = "\n".join(lines[-lines_count:])
                        elif content_type == "summary":
                            processed_content = f"File has {len(lines)} lines, {len(content)} characters"
                        else:
                            processed_content = content
                        
                        return {
                            "success": True,
                            "message": "File content read successfully",
                            "filename": filename,
                            "content_type": content_type,
                            "content": processed_content,
                            "file_info": {
                                "lines": len(lines),
                                "characters": len(content)
                            },
                            "contact_name": contact_name
                        }
                        
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"Failed to read file content: {str(e)}"
                    }
                finally:
                    # Clean up temp file
                    try:
                        if os.path.exists(downloaded_path):
                            os.remove(downloaded_path)
                    except Exception:
                        pass  # Ignore cleanup errors
                    
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to read file: {str(e)}"
                }
            finally:
                if client:
                    await client.disconnect()
        
        return self.telegram_base._run_async(_read_file_content())


class TelegramToolkit(Toolkit):
    """
    Complete Telegram toolkit containing all available tools.
    """
    
    def __init__(self, api_id: str = None, api_hash: str = None, phone: str = None, name: str = "TelegramToolkit"):
        """
        Initialize the Telegram toolkit.
        
        Args:
            api_id (str, optional): Telegram API ID. If not provided, will try to get from TELEGRAM_API_ID environment variable.
            api_hash (str, optional): Telegram API Hash. If not provided, will try to get from TELEGRAM_API_HASH environment variable.
            phone (str, optional): Phone number for authentication. If not provided, will try to get from TELEGRAM_PHONE environment variable.
            name (str): Toolkit name
        """
        # Create shared Telegram base instance
        telegram_base = TelegramBase(api_id=api_id, api_hash=api_hash, phone=phone)
        
        # Create all tools with shared base
        tools = [
            FetchLatestMessagesTool(telegram_base=telegram_base),
            SearchMessagesByKeywordTool(telegram_base=telegram_base),
            SendMessageTool(telegram_base=telegram_base),  # This is now send_message_by_name
            ListRecentChatsTool(telegram_base=telegram_base),
            FindAndRetrieveFileTool(telegram_base=telegram_base),
            SummarizeContactMessagesTool(telegram_base=telegram_base),
            DownloadFileTool(telegram_base=telegram_base),
            ReadFileContentTool(telegram_base=telegram_base)
        ]
        
        # Initialize parent with tools
        super().__init__(name=name, tools=tools)
        
        # Store base instance for access
        self.telegram_base = telegram_base