import os
import re
import json
import argparse
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from spacy.matcher import PhraseMatcher
import numpy as np

# Download necessary NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Load spaCy model for entity recognition
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("Downloading spaCy model...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process text files and match them to hierarchy items.')
    parser.add_argument('--directory', type=str, required=True, help='Directory containing text files to process')
    parser.add_argument('--json_file', type=str, required=False, default=None, help='JSON file with hierarchy data (optional)')
    parser.add_argument('--output', type=str, required=False, default='enhanced_hierarchy.json', help='Output JSON file name')
    parser.add_argument('--threshold', type=float, required=False, default=0.6, help='Confidence threshold for matches (0-1)')
    return parser.parse_args()

def extract_url_and_content(file_path):
    """Extract URL and main content from a text file"""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Extract URL using regex
    url_match = re.search(r'URL: (https?://\S+)', content)
    url = url_match.group(1) if url_match else None
    
    # Extract title for additional context
    title_match = re.search(r'Title: (.*)', content)
    title = title_match.group(1) if title_match else os.path.basename(file_path)
    
    # Extract main content after separator line
    parts = content.split('==============================\n', 1)
    main_content = parts[1] if len(parts) > 1 else content
    
    return {
        'url': url,
        'title': title,
        'content': main_content
    }

def get_wordnet_pos(tag):
    """Map POS tag to WordNet POS tag"""
    tag_dict = {
        'J': wordnet.ADJ,
        'N': wordnet.NOUN,
        'V': wordnet.VERB,
        'R': wordnet.ADV
    }
    return tag_dict.get(tag[0].upper(), wordnet.NOUN)

def preprocess_text(text):
    """Preprocess text for analysis"""
    # Lowercase and tokenize
    words = word_tokenize(text.lower())
    
    # Remove stopwords and non-alphabetic tokens
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.isalpha() and word not in stop_words]
    
    # Lemmatize words based on POS
    lemmatizer = WordNetLemmatizer()
    pos_tags = nltk.pos_tag(words)
    words = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tags]
    
    return words

def extract_phrases_with_context(text, n=3):
    """Extract n-grams with positional context"""
    tokens = word_tokenize(text.lower())
    phrases = []
    
    # Create n-grams (1, 2, and 3) with position information
    for i in range(len(tokens)):
        # Unigrams
        phrases.append((tokens[i], i/len(tokens)))
        
        # Bigrams
        if i < len(tokens) - 1:
            phrases.append((" ".join(tokens[i:i+2]), i/len(tokens)))
            
        # Trigrams
        if i < len(tokens) - 2:
            phrases.append((" ".join(tokens[i:i+3]), i/len(tokens)))
    
    return phrases

def extract_entities(text):
    """Extract named entities using spaCy"""
    doc = nlp(text)
    entities = [(ent.text.lower(), ent.label_, ent.start_char/len(text)) for ent in doc.ents]
    return entities

def get_tfidf_scores(documents):
    """Calculate TF-IDF scores for terms in documents"""
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()
    
    # Get scores for each document
    tfidf_scores = []
    for doc_idx in range(len(documents)):
        doc_scores = {}
        doc_vector = tfidf_matrix[doc_idx]
        
        for term_idx, score in zip(doc_vector.indices, doc_vector.data):
            term = feature_names[term_idx]
            doc_scores[term] = score
        
        tfidf_scores.append(doc_scores)
    
    return tfidf_scores

def extract_hierarchy_nodes(node, result=None, path=None, synonyms=None):
    """Extract all nodes from hierarchy with path information"""
    if result is None:
        result = {}
    if path is None:
        path = []
    if synonyms is None:
        synonyms = defaultdict(set)
        result['_synonyms'] = synonyms
    
    current_path = path + [node['name']]
    node_key = node['name'].lower()
    
    # Store node with its path and initialize url list
    if node_key not in result:
        result[node_key] = {
            'name': node['name'],
            'paths': [],
            'urls': [],
            'matched_documents': []
        }
    
    result[node_key]['paths'].append(current_path)
    
    # Add synonyms and related terms
    synonyms[node_key].add(node_key)
    
    # Add simple variations
    for word in node_key.split():
        if len(word) > 3:  # Only consider meaningful words
            synonyms[node_key].add(word)
    
    # Special cases and known synonyms
    if 'carbon' in node_key:
        synonyms[node_key].update(['co2', 'greenhouse gas', 'emission'])
    elif 'solar' in node_key:
        synonyms[node_key].update(['photovoltaic', 'pv', 'sun'])
    elif 'wind' in node_key:
        synonyms[node_key].update(['turbine', 'windmill'])
    elif 'battery' in node_key:
        synonyms[node_key].update(['batteries', 'energy storage', 'lithium'])
    elif 'mushroom' in node_key:
        synonyms[node_key].update(['fungi', 'mycelium', 'shroom'])
    elif 'transportation' in node_key:
        synonyms[node_key].update(['transport', 'mobility', 'vehicle'])
    elif 'building' in node_key:
        synonyms[node_key].update(['construction', 'house', 'housing'])
    
    # Process children recursively
    if 'children' in node:
        for child in node['children']:
            extract_hierarchy_nodes(child, result, current_path, synonyms)
    
    return result

def create_enhanced_pattern_matcher(hierarchy_nodes):
    """Create pattern matcher using spaCy for more flexible matching"""
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    node_patterns = {}
    
    for node_key, node_info in hierarchy_nodes.items():
        if node_key == '_synonyms':
            continue
            
        # Create patterns for the node name and its synonyms
        synonyms = hierarchy_nodes['_synonyms'].get(node_key, [node_key])
        patterns = [nlp(term) for term in synonyms if len(term) > 1]
        
        if patterns:
            node_patterns[node_key] = patterns
            matcher.add(node_key, patterns)
    
    return matcher

def find_matches_in_document(doc_info, hierarchy_nodes, matcher, threshold=0.6):
    """Find matches in document with confidence scores"""
    document = doc_info['content']
    
    # For very long documents, we'll split into manageable chunks for spaCy
    max_length = 100000  # spaCy has a limit on text length
    chunks = []
    for i in range(0, len(document), max_length):
        chunks.append(document[i:i+max_length])
    
    document_spacy = [nlp(chunk) for chunk in chunks]
    title_spacy = nlp(doc_info['title'])
    
    # Prepare results
    matches = {}
    
    # 1. Use spaCy matcher for flexible matching
    spacy_matches = []
    for i, doc in enumerate(document_spacy):
        offset = i * max_length
        for match_id, start, end in matcher(doc):
            spacy_matches.append((match_id, start + offset, end + offset))
    
    spacy_title_matches = matcher(title_spacy)
    
    # Group matches by node key
    for match_id, start, end in spacy_matches + spacy_title_matches:
        node_key = nlp.vocab.strings[match_id]
        position = start / (len(document) or 1)  # Avoid division by zero
        
        # Title matches get higher weight
        is_title_match = (match_id, start, end) in [(m[0], m[1], m[2]) for m in spacy_title_matches]
        position_weight = 1.5 if is_title_match else 1.0 - 0.5 * position  # Weight by position
        
        if node_key not in matches:
            matches[node_key] = {
                'count': 0,
                'positions': [],
                'confidence': 0
            }
        
        matches[node_key]['count'] += 1
        matches[node_key]['positions'].append(position)
    
    # 2. TF-IDF for additional context
    tfidf_scores = get_tfidf_scores([document])[0]
    
    # 3. Calculate confidence scores
    for node_key, node_info in hierarchy_nodes.items():
        if node_key == '_synonyms':
            continue
            
        if node_key in matches:
            # Base confidence from match count and positions
            match_info = matches[node_key]
            count_weight = min(1.0, match_info['count'] / 10.0)  # Normalize count
            position_weight = 1.0
            
            if match_info['positions']:
                # Weight by average position, giving higher weight to matches near beginning
                avg_position = sum(match_info['positions']) / len(match_info['positions'])
                position_weight = 1.0 - 0.5 * avg_position
            
            # Check if any synonyms have good TF-IDF scores
            tfidf_weight = 0.0
            for term in hierarchy_nodes['_synonyms'].get(node_key, []):
                if term in tfidf_scores:
                    tfidf_weight = max(tfidf_weight, tfidf_scores[term])
            
            # Hierarchical context: higher confidence for more specific matches
            path_depth_weight = 1.0
            max_depth = 0
            for path in node_info['paths']:
                max_depth = max(max_depth, len(path))
            path_depth_weight = min(1.0, max_depth / 4.0)
            
            # Final confidence score (weighted combination)
            confidence = (0.4 * count_weight + 
                          0.3 * position_weight + 
                          0.2 * tfidf_weight +
                          0.1 * path_depth_weight)
            
            matches[node_key]['confidence'] = confidence
    
    # Filter by threshold and sort by confidence
    valid_matches = {
        node_key: {
            'name': hierarchy_nodes[node_key]['name'],
            'confidence': match_info['confidence'],
            'count': match_info['count']
        }
        for node_key, match_info in matches.items()
        if match_info['confidence'] >= threshold
    }
    
    # Sort by confidence
    sorted_matches = sorted(
        valid_matches.items(), 
        key=lambda x: x[1]['confidence'], 
        reverse=True
    )
    
    return sorted_matches

def assign_urls_to_hierarchy(hierarchy, matches_by_document):
    """Assign URLs to hierarchy nodes based on document matches"""
    def update_node_urls(node, matches):
        node_key = node['name'].lower()
        
        # Check for matches in all documents
        for doc_id, doc_matches in matches.items():
            for match_key, match_info in doc_matches:
                if match_key == node_key and doc_id.startswith('http'):
                    # Add URL if not already added
                    if 'urls' not in node:
                        node['urls'] = []
                    
                    # Add URL with confidence score
                    url_entry = {
                        'url': doc_id,
                        'confidence': match_info['confidence']
                    }
                    
                    # Avoid duplicates
                    if not any(entry['url'] == doc_id for entry in node.get('urls', [])):
                        node['urls'].append(url_entry)
        
        # Process children recursively
        if 'children' in node:
            for child in node['children']:
                update_node_urls(child, matches)
    
    # Start recursive URL assignment
    update_node_urls(hierarchy, matches_by_document)
    return hierarchy

def process_directory(directory, hierarchy=None, output_file='enhanced_hierarchy.json', threshold=0.6):
    """Process all text files in a directory and match them to hierarchy items"""
    # Load hierarchy if provided, or use default structure
    if hierarchy is None:
        hierarchy_data = {
            "name": "Climate Drift",
            "children": [
                {
                    "name": "Climate Solutions",
                    "children": []
                }
            ]
        }
    else:
        if isinstance(hierarchy, str):
            with open(hierarchy, 'r') as f:
                hierarchy_data = json.load(f)
        else:
            hierarchy_data = hierarchy
    
    # Extract all nodes from hierarchy with synonyms
    hierarchy_nodes = extract_hierarchy_nodes(hierarchy_data)
    
    # Create pattern matcher
    matcher = create_enhanced_pattern_matcher(hierarchy_nodes)
    
    # Process each file in the directory
    matches_by_document = {}
    document_contents = {}
    
    print(f"Processing files in {directory}...")
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            print(f"Processing {filename}...")
            
            # Extract URL and content
            doc_info = extract_url_and_content(file_path)
            if not doc_info['url']:
                print(f"Warning: No URL found in {filename}")
                continue
            
            # Find matches in document
            matches = find_matches_in_document(doc_info, hierarchy_nodes, matcher, threshold)
            
            # Store matches by document URL
            matches_by_document[doc_info['url']] = matches
            
            # Store document info for reporting
            document_contents[doc_info['url']] = {
                'filename': filename,
                'title': doc_info['title']
            }
    
    # Assign URLs to hierarchy nodes
    enhanced_hierarchy = assign_urls_to_hierarchy(hierarchy_data, matches_by_document)
    
    # Add clickable URLs to the HTML visualization
    prepare_hierarchy_for_visualization(enhanced_hierarchy)
    
    # Save enhanced hierarchy to file
    with open(output_file, 'w') as f:
        json.dump(enhanced_hierarchy, f, indent=2)
    
    print(f"Enhanced hierarchy saved to {output_file}")
    
    # Generate report
    print("\nMatching Report:")
    for doc_url, matches in matches_by_document.items():
        if doc_url in document_contents:
            print(f"\nDocument: {document_contents[doc_url]['filename']}")
            print(f"Title: {document_contents[doc_url]['title']}")
            print(f"URL: {doc_url}")
            print("Matches:")
            
            for match_key, match_info in matches:
                print(f"  - {match_info['name']} (confidence: {match_info['confidence']:.2f}, count: {match_info['count']})")
    
    return enhanced_hierarchy

def prepare_hierarchy_for_visualization(hierarchy):
    """Prepare hierarchy data for D3.js visualization with clickable URLs"""
    def add_url_to_node(node):
        # If node has urls, prepare them for visualization
        if 'urls' in node and node['urls']:
            # Sort by confidence and take the most confident matches
            sorted_urls = sorted(node['urls'], key=lambda x: x['confidence'], reverse=True)
            
            # Add top URLs to node for visualization
            node['url_data'] = sorted_urls[:3]  # Limit to top 3 to avoid cluttering
            
        # Process children recursively
        if 'children' in node:
            for child in node['children']:
                add_url_to_node(child)
    
    # Start recursive processing
    add_url_to_node(hierarchy)
    return hierarchy

def main():
    args = parse_arguments()
    process_directory(
        args.directory, 
        args.json_file, 
        args.output,
        args.threshold
    )

if __name__ == "__main__":
    main()