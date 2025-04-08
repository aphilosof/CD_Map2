import PyPDF2
import re
import json
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Download NLTK resources if you haven't already
# nltk.download('punkt')

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def identify_hierarchy_level(line):
    """
    Determine the hierarchy level based on indentation or other markers.
    This is a simplified example - you'll need to adapt this to your PDF's format.
    """
    # For a taxonomy list like in your PDF, we might detect hierarchy by 
    # looking at indentation (which is lost in plain text) or by looking for 
    # bullets, numbers, or other markers
    
    # This is a placeholder logic - you'll need to customize this
    if re.match(r'^\s*•\s+', line):  # Bullet points
        return 1
    elif re.match(r'^\s{4,}', line):  # Indented with 4+ spaces
        return 2
    elif re.match(r'^\s{2,}', line):  # Indented with 2+ spaces
        return 1
    else:
        return 0  # Top level

def build_hierarchy(lines):
    """
    Build a hierarchical structure from the lines of text.
    Returns a nested dict representing the hierarchy.
    """
    result = {"name": "Climate Drift", "children": []}
    stack = [result]
    
    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue
            
        # Determine the level of this item
        level = identify_hierarchy_level(line)
        
        # Clean up the line text
        clean_text = line.strip().replace('•', '').strip()
        
        # Create a new node
        new_node = {"name": clean_text}
        
        # Ensure the stack has enough elements for this level
        while len(stack) > level + 1:
            stack.pop()
            
        # Add children array if it doesn't exist
        if "children" not in stack[level]:
            stack[level]["children"] = []
            
        # Add the new node to its parent's children
        stack[level]["children"].append(new_node)
        
        # Push this node onto the stack so its children can be added to it
        stack.append(new_node)
    
    return result

def parse_pdf_to_json(pdf_path, output_path):
    """
    Parse a PDF containing a hierarchical taxonomy into a JSON file.
    """
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    
    # Split into lines, removing empty lines
    lines = [line for line in text.split('\n') if line.strip()]
    
    # Build the hierarchy
    hierarchy = build_hierarchy(lines)
    
    # Write to JSON file
    with open(output_path, 'w') as file:
        json.dump(hierarchy, file, indent=2)
    
    return hierarchy

# Example usage
if __name__ == "__main__":
    pdf_path = "climate_drift.pdf"
    output_path = "climate_drift_taxonomy.json"
    hierarchy = parse_pdf_to_json(pdf_path, output_path)
    print(f"JSON hierarchy written to {output_path}")