import os
import json
from langchain.schema import Document
import pickle

# Path to the folder containing the JSON files
mapping_folder = "./sources/mapping"

# Initialize a list to store all the Document objects
all_documents = []
processed_contents = set()  # To keep track of unique content


def is_valid_content(content):
    """
    Check if the content is valid (not empty and not duplicated).
    """
    if not content.strip():  # Check for empty content
        return False
    if content in processed_contents:  # Check for duplicates
        return False
    return True

# Iterate over all files in the folder
for filename in os.listdir(mapping_folder):
    # Process only JSON files
    if filename.endswith(".json"):
        file_path = os.path.join(mapping_folder, filename)

        # Read the JSON file
        with open(file_path, "r", encoding="utf-8") as file:
            try:
                data = json.load(file)  # Assuming each JSON is a list of dictionaries
                print(f"Processing file: {filename}, Total entries: {len(data)}")

                # Transform each dictionary into a Document
                for entry in data:
                    content = entry.get("content", "")

                    if is_valid_content(content):
                        metadata = {
                            "org_path": entry.get("org_path", ""),
                            "url": entry.get("url", ""),
                            "hyperlinks": entry.get("hyperlinks", []),
                            "tables": entry.get("tables", []),
                            "txt_path": entry.get("txt_path", "")
                        }
                        document = Document(page_content=content, metadata=metadata)
                        all_documents.append(document)
                        processed_contents.add(content)  # Add to the set of processed contents

            except json.JSONDecodeError:
                print(f"Failed to decode JSON in file: {file_path}")

# Output the total number of documents created
print(f"Total valid documents created: {len(all_documents)}")

# Path to save the documents
pickle_file = "./sources/documents/doc_summary.pkl"

# Save documents as a pickle file
with open(pickle_file, "wb") as file:
    pickle.dump(all_documents, file)

print(f"Documents saved to {pickle_file}")
