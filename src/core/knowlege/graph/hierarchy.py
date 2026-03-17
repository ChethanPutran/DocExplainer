from src.core.document import DocumentChunk, DocumentNode, DocumentTree

def build_document_hierarchy(document_text: str, title: str = "") -> DocumentTree:
    """Build document hierarchy from text"""
    section_texts = document_text.split("\n\n")

    full_doc_chunk = DocumentChunk(text=document_text)
    root_node = DocumentNode(id=0, chunk=full_doc_chunk)
    doc_tree = DocumentTree(title=title or "Untitled Document", root=root_node)

    hierarchy = {
        "document": [full_doc_chunk],
        "sections": [],
        "paragraphs": [],
        "sentences": [],
    }

    for s_idx, s_text in enumerate(section_texts):
        s_chunk = DocumentChunk(text=s_text)
        hierarchy["sections"].append(s_chunk)
        root_node.children[s_idx] = DocumentNode(s_idx, s_chunk)
        parent = root_node.children[s_idx]

        paragraphs = s_text.split("\n")
        for p_idx, p_text in enumerate(paragraphs):
            if not p_text.strip():
                continue
                
            p_chunk = DocumentChunk(text=p_text)
            parent.children[p_idx] = DocumentNode(p_idx, p_chunk)
            hierarchy["paragraphs"].append(p_chunk)
            
            # Split paragraph into sentences (simple split)
            sentences = [s.strip() for s in p_text.split('.') if s.strip()]
            for sent_idx, sent_text in enumerate(sentences):
                sent_chunk = DocumentChunk(text=sent_text + '.')
                parent.children[p_idx].children[sent_idx] = DocumentNode(sent_idx, sent_chunk)
                hierarchy["sentences"].append(sent_chunk)

    doc_tree.hierarchy = hierarchy
    return doc_tree