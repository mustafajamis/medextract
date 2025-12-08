"""
Enhanced Medical Data Extraction Algorithms
Improved rule-based and hybrid extraction strategies
"""
import re
from typing import Optional, Dict, List, Tuple
import spacy

class EnhancedMedicalExtractor:
    """
    Advanced rule-based extractor with context awareness, negation handling,
    and domain-specific pattern recognition for medical severity scores.
    """
    
    def __init__(self, valid_values: List[str], target_variable: str):
        """
        Initialize extractor with valid values and target variable name.
        
        Args:
            valid_values: List of valid severity codes (e.g., ['0', '1', '1a', '2', 'NR'])
            target_variable: Name of the variable to extract (e.g., 'Symptom Severity')
        """
        self.valid_values = set(v.lower() for v in valid_values if v != 'NR')
        self.target_variable = target_variable.lower()
        
        # Try to load spaCy for better NLP (fallback to regex if unavailable)
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.use_spacy = True
        except:
            self.nlp = None
            self.use_spacy = False
    
    def extract(self, text: str) -> str:
        """
        Main extraction method with multi-strategy approach.
        
        Strategies (in order of priority):
        1. Explicit structured mentions (e.g., "Score: 2a")
        2. Direct label statements (e.g., "Condition is 1b")
        3. Contextual keyword matching with negation handling
        4. Severity inference from descriptors
        """
        if not text or not text.strip():
            return "NR"
        
        text_lower = text.lower()
        
        # STRATEGY 1: Explicit structured mentions (highest confidence)
        result = self._extract_structured_mention(text_lower)
        if result:
            return result
        
        # STRATEGY 2: Direct label statements
        result = self._extract_direct_statement(text_lower)
        if result:
            return result
        
        # STRATEGY 3: Contextual extraction with negation handling
        result = self._extract_with_context(text, text_lower)
        if result:
            return result
        
        # STRATEGY 4: Severity descriptor inference
        result = self._infer_from_descriptors(text_lower)
        if result:
            return result
        
        return "NR"
    
    def _extract_structured_mention(self, text: str) -> Optional[str]:
        """
        Extract from structured patterns like:
        - "Score: 2a"
        - "Severity level = 1b"
        - "BTFU Score (Updated): 3"
        - "[Symptom Severity: 0]"
        """
        # Pattern: variable name followed by score
        variable_keywords = [
            'score', 'severity', 'level', 'grade', 'rating', 
            'classification', 'btfu', 'symptom'
        ]
        
        for keyword in variable_keywords:
            # Match patterns like "keyword: 2a" or "keyword = 1b"
            pattern = rf'\b{keyword}\s*(?:score|level|rating)?\s*[:=\-]\s*([0-4](?:[abc])?|nr)\b'
            match = re.search(pattern, text)
            if match:
                value = match.group(1).lower()
                if value == 'nr' or value in self.valid_values:
                    return value.upper() if value != 'nr' else 'NR'
        
        # Direct structured statement: "Condition is 1b"
        pattern = r'\b(?:condition|status|state|severity)\s+(?:is|:)\s*([0-4](?:[abc])?|nr)\b'
        match = re.search(pattern, text)
        if match:
            value = match.group(1).lower()
            if value == 'nr' or value in self.valid_values:
                return value.upper() if value != 'nr' else 'NR'
        
        return None
    
    def _extract_direct_statement(self, text: str) -> Optional[str]:
        """
        Extract from direct statements like:
        - "The patient has a severity of 2a"
        - "Classified as 1b"
        - "Score indicates 3a"
        """
        patterns = [
            r'severity\s+(?:of|is|:)?\s*([0-4](?:[abc])?)\b',
            r'classified\s+as\s+([0-4](?:[abc])?)\b',
            r'categorized\s+as\s+([0-4](?:[abc])?)\b',
            r'rated\s+(?:as\s+)?([0-4](?:[abc])?)\b',
            r'grade\s+([0-4](?:[abc])?)\b',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                value = match.group(1).lower()
                if value in self.valid_values:
                    return value.upper()
        
        return None
    
    def _extract_with_context(self, text: str, text_lower: str) -> Optional[str]:
        """
        Context-aware extraction with negation handling using dependency parsing.
        """
        # Negation indicators
        negation_words = {
            'no', 'not', 'without', 'absence', 'lack', 'free', 
            'denies', 'negative', 'resolved', 'absent'
        }
        
        # Create mapping of severity descriptors to scores
        severity_map = self._build_severity_map()
        
        if self.use_spacy and self.nlp:
            doc = self.nlp(text)
            
            # Look for severity descriptors with dependency parsing
            for token in doc:
                if token.text.lower() in severity_map:
                    # Check if negated
                    is_negated = any(
                        child.text.lower() in negation_words 
                        for child in token.children
                    )
                    is_negated = is_negated or any(
                        ancestor.text.lower() in negation_words
                        for ancestor in token.ancestors
                    )
                    
                    if not is_negated:
                        # Check temporal context - prefer current state
                        is_past = any(
                            child.dep_ == 'advmod' and 
                            child.text.lower() in ['previously', 'formerly', 'past', 'before']
                            for child in token.children
                        )
                        
                        if not is_past:
                            return severity_map[token.text.lower()]
        
        # Fallback: Regex-based context checking
        return self._extract_with_regex_context(text_lower, severity_map, negation_words)
    
    def _extract_with_regex_context(self, text: str, severity_map: Dict, 
                                     negation_words: set) -> Optional[str]:
        """
        Fallback context extraction using regex when spaCy is unavailable.
        """
        for descriptor, score in severity_map.items():
            # Find descriptor with context window
            pattern = rf'(.{{0,50}})\b{re.escape(descriptor)}\b(.{{0,50}})'
            match = re.search(pattern, text)
            
            if match:
                context_before = match.group(1)
                context_after = match.group(2)
                full_context = context_before + descriptor + context_after
                
                # Check for negation in context
                has_negation = any(
                    neg in full_context.split() 
                    for neg in negation_words
                )
                
                # Check for past tense indicators
                is_past = any(
                    word in full_context 
                    for word in ['previously', 'formerly', 'was', 'had been', 'past']
                )
                
                if not has_negation and not is_past:
                    return score
        
        return None
    
    def _infer_from_descriptors(self, text: str) -> Optional[str]:
        """
        Infer severity from clinical descriptors when explicit scores are absent.
        Uses weighted scoring based on multiple indicators.
        """
        # Severity indicators with scores
        indicators = {
            # Level 0 indicators
            'asymptomatic': 0,
            'no symptoms': 0,
            'stable': 0,
            'normal': 0,
            'routine': 0,
            'resolved': 0,
            
            # Level 1 indicators
            'mild': 1,
            'minor': 1,
            'slight': 1,
            'minimal': 1,
            'improving': 1,
            
            # Level 2 indicators
            'moderate': 2,
            'intermediate': 2,
            'adjustment': 2,
            'medication change': 2,
            
            # Level 3 indicators
            'severe': 3,
            'serious': 3,
            'significant': 3,
            'worsening': 3,
            'deteriorated': 3,
            
            # Level 4 indicators
            'critical': 4,
            'emergency': 4,
            'icu': 4,
            'life-threatening': 4,
            'acute distress': 4,
        }
        
        # Count occurrences and calculate weighted score
        score_counts = {}
        for phrase, level in indicators.items():
            if phrase in text:
                score_counts[level] = score_counts.get(level, 0) + 1
        
        if not score_counts:
            return None
        
        # Return the most frequently occurring level
        max_level = max(score_counts.items(), key=lambda x: x[1])[0]
        
        # Convert numeric level to string format
        return str(max_level)
    
    def _build_severity_map(self) -> Dict[str, str]:
        """Build mapping of clinical terms to severity scores."""
        severity_map = {
            # Level 0
            'asymptomatic': '0',
            'resolved': '0',
            'normal': '0',
            'stable': '0',
            
            # Level 1
            'mild': '1',
            'minor': '1',
            'slight': '1',
            
            # Level 2  
            'moderate': '2',
            'intermediate': '2',
            
            # Level 3
            'severe': '3',
            'serious': '3',
            'significant': '3',
            
            # Level 4
            'critical': '4',
            'emergency': '4',
            'acute': '4',
        }
        
        return severity_map


class HybridExtractor:
    """
    Combines rule-based extraction with LLM-based extraction.
    Uses confidence scoring to determine which method to trust.
    """
    
    def __init__(self, rule_extractor: EnhancedMedicalExtractor):
        self.rule_extractor = rule_extractor
    
    def extract_with_confidence(self, text: str, llm_result: Optional[str]) -> Tuple[str, float]:
        """
        Extract value with confidence score.
        
        Returns:
            (extracted_value, confidence_score)
        """
        rule_result = self.rule_extractor.extract(text)
        
        # If both agree, high confidence
        if llm_result and rule_result != "NR" and llm_result == rule_result:
            return rule_result, 0.95
        
        # If rule-based found structured mention, trust it over LLM
        if rule_result != "NR":
            structured_result = self.rule_extractor._extract_structured_mention(text.lower())
            if structured_result:
                return structured_result, 0.90
        
        # If LLM found something and rule-based didn't, moderate confidence
        if llm_result and llm_result != "NR" and rule_result == "NR":
            return llm_result, 0.70
        
        # If only rule-based found something, moderate confidence
        if rule_result != "NR" and (not llm_result or llm_result == "NR"):
            return rule_result, 0.65
        
        # Both returned NR or disagreed with no clear winner
        if rule_result == "NR" and (not llm_result or llm_result == "NR"):
            return "NR", 0.80  # High confidence in NR when both agree
        
        # Disagreement - prefer rule-based for structured data
        return rule_result, 0.50


# Utility function for testing
def test_extractor():
    """Test the enhanced extractor with sample medical reports."""
    
    test_cases = [
        ("Patient presents with mild symptoms. No complications noted.", "1"),
        ("Condition is 1b. Patient prescribed new medication.", "1b"),
        ("Score: 2a. Moderate symptoms requiring adjustment.", "2a"),
        ("No severe symptoms. Patient stable and asymptomatic.", "0"),
        ("Previously severe, now mild symptoms after treatment.", "1"),
        ("Critical condition. ICU admission required.", "4"),
    ]
    
    extractor = EnhancedMedicalExtractor(
        valid_values=['0', '1', '1a', '1b', '2', '2a', '3', '3a', '4', 'NR'],
        target_variable='Symptom Severity'
    )
    
    print("Testing Enhanced Extractor:")
    print("-" * 80)
    for text, expected in test_cases:
        result = extractor.extract(text)
        status = "✓" if result == expected else "✗"
        print(f"{status} Expected: {expected:3} | Got: {result:3} | Text: {text[:60]}...")
    print("-" * 80)


if __name__ == "__main__":
    test_extractor()
