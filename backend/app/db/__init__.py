"""
Database package for persistent SQLite storage of chats, documents, and messages.
"""

from app.db.database import get_db_connection, init_db
from app.db.repository import (
    create_chat,
    list_chats,
    get_chat,
    rename_chat,
    delete_chat,
    create_document,
    get_document_by_id,
    get_document_by_hash,
    update_document_status,
    list_documents,
    attach_document_to_chat,
    detach_document_from_chat,
    list_chat_documents,
    document_attached_to_chat,
    create_message,
    list_messages,
)

__all__ = [
    "get_db_connection",
    "init_db",
    "create_chat",
    "list_chats",
    "get_chat",
    "rename_chat",
    "delete_chat",
    "create_document",
    "get_document_by_id",
    "get_document_by_hash",
    "update_document_status",
    "list_documents",
    "attach_document_to_chat",
    "detach_document_from_chat",
    "list_chat_documents",
    "document_attached_to_chat",
    "create_message",
    "list_messages",
]
