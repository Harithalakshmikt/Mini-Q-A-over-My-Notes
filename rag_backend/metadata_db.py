
from rag_backend.metadata_db import SQLiteMetadataStore  # Local

from rag_backend.metadata_db import DynamoDBMetadataStore  # AWS

metadata_store = SQLiteMetadataStore("./metadata.db")
metadata_store.save_retrieval(chunk_id, question, score)