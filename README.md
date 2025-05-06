# CS2540 Project: Adversarial Examples in Natural Language Processing

This project explores the robustness of BERT-based sentiment classification models against adversarial attacks. It implements and compares two different approaches to finding adversarial examples: SMT (Satisfiability Modulo Theories) and FGSM (Fast Gradient Sign Method).

## Project Structure

```
cs2540proj/
├── bert/                    # BERT-based sentiment analysis experiments
├── cm_2/                    # Circles and Moons classification
├── iris_flower/            # Iris flower classification
├── circles_moons_classification/
├── pass_fail/              # Pass/Fail classification
├── adnet_main/             # Adversarial Network experiments
├── xor/                    # XOR classification
└── requirements.txt        # Project dependencies
```

## Setup and Installation

1. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Unix/macOS
# OR
.\venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running Experiments

1. Run BERT experiments:
```bash
cd bert
python same_perturbed.py
```

2. Analyze results:
```bash
python analyze_overlap.py
```

## Key Components

1. **SMT Verification**
   - Uses Z3 solver to find minimal perturbations
   - Currently implements case changes and minor modifications
   - More conservative approach to finding adversarial examples

2. **FGSM Attacks**
   - Uses gradient-based approach to find perturbations
   - Makes more aggressive changes to the input
   - Can completely replace words with different ones

3. **Analysis Tools**
   - Visualizes token-level changes
   - Compares effectiveness of both methods

## Results

The experiments show:
1. FGSM is more successful at finding adversarial examples
2. SMT makes more conservative, semantic-preserving changes
3. The overlap between perturbations from both methods is low
4. The model shows varying levels of robustness to different types of attacks

## Other Experiments

The repository also contains several other classification experiments:
- Circles and Moons classification
- Iris flower classification
- Pass/Fail classification
- XOR classification
- Adversarial Network experiments

Each experiment demonstrates different aspects of machine learning robustness and adversarial attacks.

## Dependencies

Key dependencies include:
- PyTorch
- Transformers (BERT)
- Z3 Solver
- NumPy
- Matplotlib

See `requirements.txt` for complete list.

## Future Work

1. Improve SMT perturbations to make more meaningful changes
2. Enhance FGSM to preserve semantic meaning
3. Implement additional verification methods
4. Add more sophisticated visualization tools
5. Improve model robustness against adversarial attacks
