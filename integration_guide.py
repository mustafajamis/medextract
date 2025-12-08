"""
Integration Guide: Enhanced Extractor with Existing MedExtract System

This module provides seamless integration of the enhanced rule-based extractor
with the existing medextract.py pipeline.
"""

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from enhanced_extractor import EnhancedMedicalExtractor, HybridExtractor
import json
from typing import Optional


class IntegratedExtractor:
    """
    Wrapper class to integrate enhanced extraction into existing medextract workflow.
    
    Usage in medextract.py:
        from integration_guide import IntegratedExtractor
        
        # In your config loading section:
        extractor = IntegratedExtractor(config)
        
        # Replace existing process_text calls:
        result = extractor.process_text_enhanced(
            text=report_text,
            llm_model=llm_model,
            # ... other params
        )
    """
    
    def __init__(self, config: dict):
        """
        Initialize integrated extractor with configuration.
        
        Args:
            config: Configuration dictionary from load_config()
        """
        self.config = config
        
        # Initialize enhanced rule-based extractor
        self.rule_extractor = EnhancedMedicalExtractor(
            valid_values=config['evaluation']['valid_values'],
            target_variable=config['evaluation']['target_variable']
        )
        
        # Initialize hybrid extractor
        self.hybrid_extractor = HybridExtractor(self.rule_extractor)
        
        # Extraction mode configuration
        self.use_hybrid = config.get('extraction', {}).get('use_hybrid', True)
        self.confidence_threshold = config.get('evaluation', {}).get('confidence_threshold', 0.6)
    
    def process_text_enhanced(
        self,
        text: str,
        llm_result: Optional[str] = None,
        return_confidence: bool = False
    ) -> str:
        """
        Enhanced text processing with hybrid extraction.
        
        Args:
            text: Medical report text to process
            llm_result: Optional LLM extraction result (if already computed)
            return_confidence: Whether to include confidence score in result
        
        Returns:
            JSON string with extracted value and optionally confidence score
        """
        target_var = self.config['evaluation']['target_variable']
        
        # Parse LLM result if provided
        llm_value = None
        if llm_result:
            llm_value = self._parse_llm_result(llm_result)
        
        # Use hybrid extraction
        if self.use_hybrid:
            extracted_value, confidence = self.hybrid_extractor.extract_with_confidence(
                text=text,
                llm_result=llm_value
            )
            
            if return_confidence:
                return json.dumps({
                    target_var: extracted_value,
                    "confidence": round(confidence, 3)
                })
            else:
                return json.dumps({target_var: extracted_value})
        
        # Fallback to rule-based only
        else:
            extracted_value = self.rule_extractor.extract(text)
            return json.dumps({target_var: extracted_value})
    
    def _parse_llm_result(self, llm_result: str) -> Optional[str]:
        """
        Parse LLM JSON result to extract value.
        
        Args:
            llm_result: JSON string from LLM
        
        Returns:
            Extracted value or None if parsing fails
        """
        try:
            # Handle various JSON formats
            result_dict = json.loads(llm_result)
            target_var = self.config['evaluation']['target_variable']
            return result_dict.get(target_var)
        except:
            # If JSON parsing fails, try direct string matching
            valid_values = self.config['evaluation']['valid_values']
            for value in valid_values:
                if value.lower() in llm_result.lower():
                    return value
            return None


# ============================================================================
# INTEGRATION INSTRUCTIONS
# ============================================================================

def integration_example():
    """
    Example showing how to modify medextract.py to use enhanced extraction.
    """
    
    print("""
    === INTEGRATION STEPS ===
    
    Step 1: Import the integrated extractor at the top of medextract.py
    -----------------------------------------------------------------------
    Add after existing imports:
    
    from integration_guide import IntegratedExtractor
    
    
    Step 2: Initialize in the main() function
    -----------------------------------------------------------------------
    After loading config, add:
    
    # Initialize enhanced extractor (add after config loading)
    integrated_extractor = IntegratedExtractor(config)
    
    
    Step 3: Modify process_text() function
    -----------------------------------------------------------------------
    Replace the existing function with this enhanced version:
    
    def process_text(text, llm_model, rag_enabled, embeddings, retriever_type, 
                     reranker, simple_prompting, fewshots_method, 
                     fewshots_with_NR_method, fewshots_with_NR_extended_method, 
                     json_value, temp, top_k, top_p):
        
        # Check if force_rule_based is enabled
        force_rule = config.get('processing', {}).get('force_rule_based', False)
        
        if force_rule:
            # Use enhanced rule-based only
            return integrated_extractor.process_text_enhanced(
                text=text,
                return_confidence=True
            )
        
        # Otherwise, run normal LLM process first
        if rag_enabled:
            chunks = get_text_chunks(
                text, 
                chunk_size=config['rag']['chunk_size'], 
                chunk_overlap=config['rag']['chunk_overlap']
            )
            # ... rest of RAG logic ...
            
        # Get LLM result
        try:
            llm_result = ollama_llm(
                context=formatted_context,
                llm_model=llm_model,
                # ... other params ...
            )
        except Exception:
            llm_result = None
        
        # Use hybrid extraction with LLM result
        return integrated_extractor.process_text_enhanced(
            text=text if not rag_enabled else formatted_context,
            llm_result=llm_result,
            return_confidence=True
        )
    
    
    Step 4: Update config.yaml
    -----------------------------------------------------------------------
    Add to config.yaml:
    
    # Extraction configuration
    extraction:
      use_hybrid: true           # Enable hybrid extraction
      use_enhanced_rules: true   # Use enhanced rule-based extractor
      
    evaluation:
      confidence_threshold: 0.6  # Minimum confidence for predictions
    
    
    Step 5: Optional - Add confidence filtering in evaluate_experiment()
    -----------------------------------------------------------------------
    In evaluate_experiment(), after cleaning extracted values:
    
    # Filter low-confidence predictions
    if 'confidence' in df_exp[column_name].iloc[0]:
        df_exp['confidence'] = df_exp[column_name].apply(
            lambda x: json.loads(x).get('confidence', 1.0) 
            if isinstance(x, str) and x.startswith('{') else 1.0
        )
        
        # Report confidence statistics
        print(f"Confidence Statistics:")
        print(f"  Mean: {df_exp['confidence'].mean():.3f}")
        print(f"  Median: {df_exp['confidence'].median():.3f}")
        print(f"  Min: {df_exp['confidence'].min():.3f}")
        
        # Optional: Filter out low-confidence predictions
        threshold = config['evaluation'].get('confidence_threshold', 0.0)
        low_conf_count = (df_exp['confidence'] < threshold).sum()
        if low_conf_count > 0:
            print(f"  {low_conf_count} predictions below threshold {threshold}")
    
    """)


# ============================================================================
# TESTING UTILITIES
# ============================================================================

def test_integration_with_sample_data():
    """
    Test the integration with sample medical reports.
    """
    # Sample config
    test_config = {
        'evaluation': {
            'target_variable': 'Symptom Severity',
            'valid_values': ['0', '1', '1a', '1b', '2', '2a', '3', '3a', '4', 'NR'],
            'confidence_threshold': 0.6
        },
        'extraction': {
            'use_hybrid': True,
            'use_enhanced_rules': True
        }
    }
    
    # Initialize extractor
    extractor = IntegratedExtractor(test_config)
    
    # Test cases
    test_cases = [
        {
            'text': 'Patient presents with mild symptoms. No complications noted. Follow-up scheduled.',
            'llm_result': '{"Symptom Severity": "1"}',
            'expected': '1'
        },
        {
            'text': 'Minor issues noted. Patient prescribed new medication. Condition is 1b.',
            'llm_result': None,
            'expected': '1b'
        },
        {
            'text': 'Severe condition requiring immediate hospitalization.',
            'llm_result': '{"Symptom Severity": "3"}',
            'expected': '3'
        },
        {
            'text': 'Patient stable, no specific severity mentioned in this report.',
            'llm_result': None,
            'expected': 'NR'
        },
    ]
    
    print("=" * 80)
    print("INTEGRATION TEST RESULTS")
    print("=" * 80)
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        result_json = extractor.process_text_enhanced(
            text=test['text'],
            llm_result=test['llm_result'],
            return_confidence=True
        )
        
        result_dict = json.loads(result_json)
        extracted = result_dict.get('Symptom Severity')
        confidence = result_dict.get('confidence', 'N/A')
        
        status = "✓ PASS" if extracted == test['expected'] else "✗ FAIL"
        if extracted == test['expected']:
            passed += 1
        else:
            failed += 1
        
        print(f"\nTest {i}: {status}")
        print(f"  Text: {test['text'][:60]}...")
        print(f"  Expected: {test['expected']}")
        print(f"  Got: {extracted}")
        print(f"  Confidence: {confidence}")
        print(f"  LLM Input: {test['llm_result']}")
    
    print("\n" + "=" * 80)
    print(f"SUMMARY: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("=" * 80)
    
    return passed == len(test_cases)


# ============================================================================
# PERFORMANCE COMPARISON
# ============================================================================

def compare_extractors_performance():
    """
    Compare performance between original and enhanced extractors.
    """
    # Import original extractor if available
    try:
        import medextract
        original_extractor = medextract.rule_based_extractor
    except:
        print("Could not import original extractor - skipping comparison")
        return
    
    # Sample test reports
    test_reports = [
        "Patient presents with mild symptoms. No complications noted.",
        "Moderate symptoms requiring medication adjustment.",
        "Severe condition requiring immediate hospitalization.",
        "No significant symptoms reported. Patient stable.",
        "Condition is 1b. Patient prescribed new medication.",
        "Previously severe, now mild symptoms after treatment.",
        "No severe symptoms. Patient asymptomatic.",
    ]
    
    # Initialize enhanced extractor
    config = {
        'evaluation': {
            'target_variable': 'Symptom Severity',
            'valid_values': ['0', '1', '1a', '1b', '2', '2a', '3', '3a', '4', 'NR']
        }
    }
    enhanced = EnhancedMedicalExtractor(
        valid_values=config['evaluation']['valid_values'],
        target_variable=config['evaluation']['target_variable']
    )
    
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON: Original vs Enhanced")
    print("=" * 80)
    
    for i, report in enumerate(test_reports, 1):
        original_result = original_extractor(report)
        enhanced_result = enhanced.extract(report)
        
        print(f"\nTest {i}:")
        print(f"  Report: {report[:70]}...")
        print(f"  Original: {original_result}")
        print(f"  Enhanced: {enhanced_result}")
        print(f"  Match: {'✓' if original_result == enhanced_result else '✗'}")
    
    print("\n" + "=" * 80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("MEDEXTRACT INTEGRATION GUIDE")
    print("="*80)
    
    # Show integration instructions
    integration_example()
    
    # Run tests
    print("\n" + "="*80)
    print("Running Integration Tests...")
    print("="*80)
    success = test_integration_with_sample_data()
    
    if success:
        print("\n✓ All tests passed! Integration is ready.")
    else:
        print("\n✗ Some tests failed. Please review the results.")
    
    # Optional: Run performance comparison
    print("\n" + "="*80)
    print("Running Performance Comparison...")
    print("="*80)
    compare_extractors_performance()
    
    print("\n" + "="*80)
    print("Integration guide complete!")
    print("Next step: Follow the integration steps above to modify medextract.py")
    print("="*80 + "\n")
