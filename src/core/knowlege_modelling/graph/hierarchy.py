from src.core.document.document_structures import DocumentChunk, DocumentNode, DocumentTree


def build_document_hierarchy(document_text: str):
    section_texts = document_text.split("\n\n")

    full_doc_chunk = DocumentChunk(text=document_text)
    root_node = DocumentNode(id=0, chunk=full_doc_chunk)
    doc_tree = DocumentTree(title="Autonomous Systems Doc", root=root_node)

    hierarchy = {
        "document": [full_doc_chunk],
        "sections": [],
        "paragraphs": [],
        "sentences": [],
    }
    doc_tree.root = root_node

    for s_idx, s_text in enumerate(section_texts):
        s_chunk = DocumentChunk(text=s_text)
        hierarchy["sections"].append(s_chunk)
        root_node.children[s_idx] = DocumentNode(s_idx, s_chunk)
        parent = root_node.children[s_idx]

        paragraphs = s_text.split("\n")
        for p_idx, p_text in enumerate(paragraphs):
            p_chunk = DocumentChunk(text=p_text)
            parent.children[p_idx] = DocumentNode(p_idx, p_chunk)
            hierarchy["paragraphs"].append(p_chunk)
            sent_chunk = DocumentChunk(text=p_text)
            parent.children[p_idx].children[0] = DocumentNode(0, sent_chunk)
            hierarchy["sentences"].append(sent_chunk)

    doc_tree.hierarchy = hierarchy
    return doc_tree
