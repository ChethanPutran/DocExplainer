import chromadb
from pathlib import Path

path = Path("db/vector_dbs").expanduser()

client = chromadb.PersistentClient(
    path=str(path)
)

print(f"Chroma path: {path}")
print()

collections = client.list_collections()

for collection in collections:
    print(f"Collection: {collection.name}")
    print(f"Count:      {collection.count()}")

    data = collection.get(
        limit=5,
        include=["documents", "metadatas"],
    )

    for i, doc in enumerate(data["documents"]):
        print(f"\n--- Document {i} ---")
        print(doc)

        if data["metadatas"]:
            print("Metadata:", data["metadatas"][i])

    print()