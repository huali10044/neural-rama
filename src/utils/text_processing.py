"""
Text processing utilities for extracting terms and categories
Mimics Lucene + WordNet processing from paper
"""
import re
from typing import List, Set
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('stopwords')


class TextProcessor:
    """
    Extract terms (nouns) and categories from text
    Paper mentions: Lucene StandardAnalyzer + WordNet for noun extraction
    """
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        # Common words to filter
        self.common_words = {'place', 'location', 'great', 'good', 'best', 'nice'}
    
    def extract_nouns(self, text: str) -> List[str]:
        """
        Extract noun words from text (specific interests)
        Section 2.1.3: "extract noun words from title and description"
        """
        if not text:
            return []
        
        # Tokenize
        tokens = word_tokenize(text.lower())
        
        # POS tagging
        pos_tags = nltk.pos_tag(tokens)
        
        # Extract nouns (NN, NNS, NNP, NNPS)
        nouns = [
            word for word, pos in pos_tags 
            if pos.startswith('NN') and 
            word not in self.stop_words and
            word not in self.common_words and
            len(word) > 2 and
            word.isalpha()
        ]
        
        return nouns
    
    def extract_terms_simple(self, text: str) -> List[str]:
        """
        Simpler extraction: just filter meaningful words
        Useful when POS tagging is too slow
        """
        if not text:
            return []
        
        # Lowercase and tokenize
        tokens = word_tokenize(text.lower())
        
        # Filter
        terms = [
            token for token in tokens
            if token.isalpha() and
            token not in self.stop_words and
            token not in self.common_words and
            len(token) > 3
        ]
        
        return terms
    
    def is_noun(self, word: str) -> bool:
        """Check if word is a noun using WordNet"""
        synsets = wn.synsets(word)
        return any(s.pos() == 'n' for s in synsets)
    
    def extract_categories(self, poi, category_list: List[str]) -> List[str]:
        """
        Extract Yelp categories for POI
        In real TREC, this comes from Yelp API
        For synthetic data, already in POI object
        """
        return poi.categories if hasattr(poi, 'categories') else []
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        # Remove special characters
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


class TermExtractor:
    """
    Term extraction for RAMA
    Combines title and description
    """
    
    def __init__(self, use_pos_tagging: bool = True):
        self.processor = TextProcessor()
        self.use_pos_tagging = use_pos_tagging
    
    def extract_from_poi(self, poi) -> List[str]:
        """Extract terms from POI (title + description)"""
        text = f"{poi.title} {poi.description}"
        text = self.processor.clean_text(text)
        
        if self.use_pos_tagging:
            terms = self.processor.extract_nouns(text)
        else:
            terms = self.processor.extract_terms_simple(text)
        
        return terms
    
    def extract_from_candidate(self, candidate) -> List[str]:
        """Extract terms from candidate (name + snippet)"""
        text = f"{candidate.name} {candidate.snippet}"
        text = self.processor.clean_text(text)
        
        if self.use_pos_tagging:
            terms = self.processor.extract_nouns(text)
        else:
            terms = self.processor.extract_terms_simple(text)
        
        return terms


class CategoryExtractor:
    """
    Category extraction for RAMA
    """
    
    def extract_from_poi(self, poi) -> List[str]:
        """Get categories from POI"""
        return poi.categories if hasattr(poi, 'categories') else []
    
    def extract_from_candidate(self, candidate) -> List[str]:
        """Get categories from candidate"""
        return candidate.categories if hasattr(candidate, 'categories') else []


# Example usage
if __name__ == "__main__":
    processor = TextProcessor()
    
    # Test text
    sample_text = "The Art Institute of Chicago is a world-renowned museum featuring extensive collections"
    
    print("Nouns:", processor.extract_nouns(sample_text))
    print("Simple terms:", processor.extract_terms_simple(sample_text))
    
    # Test with POI
    from data_loader import PointOfInterest
    
    poi = PointOfInterest(
        poi_id="test",
        title="The Grand Museum of Natural History",
        description="Explore fascinating exhibits on dinosaurs and ancient civilizations",
        url="http://example.com",
        categories=["museums", "historical_sites"]
    )
    
    term_extractor = TermExtractor(use_pos_tagging=True)
    category_extractor = CategoryExtractor()
    
    print("\nFrom POI:")
    print("Terms:", term_extractor.extract_from_poi(poi))
    print("Categories:", category_extractor.extract_from_poi(poi))