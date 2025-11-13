UNEDITABLE_SYSTEM_PROMPT = (
    "\nIf the tool throws an error requiring authentication, provide the user with a "
    "Markdown link to the authentication page and prompt them to authenticate."
)

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that has access to a variety of tools."
)

RAG_ONLY_PROMPT = (
    "\n\n[RAG-ONLY MODE]\n"
    "You must use only the `rag_search` tool to retrieve context and answer.\n"
    "Do not call or reference any other tools, web search, or code execution.\n"
    'If the `rag_search` tool returns no relevant results, reply: "I don\'t have enough information in the knowledge base." '
    "and ask the user to add documents or refine the query.\n"
    "Do not fabricate facts beyond what the tool returns. Prefer quoting short snippets from `documents` when present.\n"
)
