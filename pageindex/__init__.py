"""
PageIndex — Markdown-based document structure indexing.

Parses .md files into hierarchical structure trees with node IDs,
optional summaries, and tree thinning for token optimization.
"""

from pageindex.page_index_md import (
    md_to_tree,
    extract_nodes_from_markdown,
    extract_node_text_content,
    build_tree_from_nodes,
    clean_tree_for_output,
)

__all__ = [
    "md_to_tree",
    "extract_nodes_from_markdown",
    "extract_node_text_content",
    "build_tree_from_nodes",
    "clean_tree_for_output",
]
