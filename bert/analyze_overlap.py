import json
import os
from collections import Counter
import numpy as np
from typing import Dict, List, Tuple

def load_results(directory: str) -> Dict:
    """Load results from the all_results.json file."""
    results_path = os.path.join(directory, 'all_results.json')
    with open(results_path, 'r') as f:
        return json.load(f)

def tokenize_text(text: str) -> List[str]:
    """Tokenize text into words, handling special characters."""
    # Convert to lowercase and split on whitespace
    tokens = text.lower().split()
    # Remove punctuation from tokens
    tokens = [token.strip('.,!?()[]{}":;') for token in tokens]
    # Remove empty tokens
    return [token for token in tokens if token]

def calculate_overlap(original: str, smt_text: str, fgsm_text: str) -> Tuple[float, Dict]:
    """Calculate overlap coefficient between SMT and FGSM perturbations."""
    # Tokenize all texts
    original_tokens = set(tokenize_text(original))
    smt_tokens = set(tokenize_text(smt_text))
    fgsm_tokens = set(tokenize_text(fgsm_text))
    
    # Find tokens that changed in each method
    smt_changes = smt_tokens - original_tokens
    fgsm_changes = fgsm_tokens - original_tokens
    
    # Calculate overlap coefficient
    if not smt_changes or not fgsm_changes:
        overlap = 0.0
    else:
        intersection = len(smt_changes.intersection(fgsm_changes))
        union = len(smt_changes.union(fgsm_changes))
        overlap = intersection / union if union > 0 else 0.0
    
    # Calculate additional metrics
    metrics = {
        'smt_changes': list(smt_changes),
        'fgsm_changes': list(fgsm_changes),
        'common_changes': list(smt_changes.intersection(fgsm_changes)),
        'smt_only_changes': list(smt_changes - fgsm_changes),
        'fgsm_only_changes': list(fgsm_changes - smt_changes),
        'total_smt_changes': len(smt_changes),
        'total_fgsm_changes': len(fgsm_changes),
        'total_common_changes': len(smt_changes.intersection(fgsm_changes))
    }
    
    return overlap, metrics

def analyze_all_examples(results: Dict) -> Dict:
    """Analyze overlap for all examples in the results."""
    analysis = {}
    total_overlap = 0.0
    successful_examples = 0
    
    for example_id, example_data in results.items():
        original_text = example_data['text']
        smt_text = example_data['results']['smt']['text']
        fgsm_text = example_data['results']['fgsm']['text']
        
        overlap, metrics = calculate_overlap(original_text, smt_text, fgsm_text)
        
        analysis[example_id] = {
            'overlap_coefficient': overlap,
            'metrics': metrics,
            'smt_success': example_data['results']['smt']['success'],
            'fgsm_success': example_data['results']['fgsm']['success']
        }
        
        if example_data['results']['smt']['success'] or example_data['results']['fgsm']['success']:
            total_overlap += overlap
            successful_examples += 1
    
    # Calculate average overlap for successful examples
    avg_overlap = total_overlap / successful_examples if successful_examples > 0 else 0.0
    
    return {
        'per_example_analysis': analysis,
        'average_overlap': avg_overlap,
        'total_examples': len(results),
        'successful_examples': successful_examples
    }

def main():
    # Find all output directories
    base_dir = 'non_robust'
    output_dirs = [d for d in os.listdir(base_dir) if d.startswith('output_')]
    
    for output_dir in output_dirs:
        full_path = os.path.join(base_dir, output_dir)
        print(f"\nAnalyzing results from {output_dir}")
        
        try:
            results = load_results(full_path)
            analysis = analyze_all_examples(results)
            
            # Print summary
            print(f"\nSummary for {output_dir}:")
            print(f"Average overlap coefficient: {analysis['average_overlap']:.3f}")
            print(f"Total examples: {analysis['total_examples']}")
            print(f"Successful examples: {analysis['successful_examples']}")
            
            # Print detailed analysis for each example
            print("\nDetailed analysis per example:")
            for example_id, example_analysis in analysis['per_example_analysis'].items():
                print(f"\n{example_id}:")
                print(f"Overlap coefficient: {example_analysis['overlap_coefficient']:.3f}")
                print(f"SMT success: {example_analysis['smt_success']}")
                print(f"FGSM success: {example_analysis['fgsm_success']}")
                print("Common changes:", example_analysis['metrics']['common_changes'])
                print("SMT-only changes:", example_analysis['metrics']['smt_only_changes'])
                print("FGSM-only changes:", example_analysis['metrics']['fgsm_only_changes'])
            
            # Save analysis to file
            output_file = os.path.join(full_path, 'overlap_analysis.json')
            with open(output_file, 'w') as f:
                json.dump(analysis, f, indent=4)
            print(f"\nAnalysis saved to {output_file}")
            
        except Exception as e:
            print(f"Error analyzing {output_dir}: {str(e)}")

if __name__ == "__main__":
    main() 
