# Thesis Appendix – LLM-based Multi-Agent sytem for M&A value prediction 

This repository contains the Python scripts and reasoning logs developed for the dissertation:

> *"Predicting Value Creation in M&A with an LLM-Based Multi-Agent System: An Empirical Analysis on European Deals"

---

## Contents

- **`/code`** – Modular Python script implementing the multi-agent architecture (Proponent, Opponent, Expert, Manager) for predicting value creation in M&A deals.
- **`/logs`** – AI reasoning logs generated for each deal, providing transparency into the deliberation process.
- **`README.md`** – Documentation and usage instructions.

---

## How to Use

1. **Clone the repository**
   ```bash
   git clone https://github.com/USERNAME/Thesis_Appendix.git
   cd Thesis_Appendix
   
2. **Set up environment**
   - Python ≥ 3.9
   - Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```
     
3. **Run the main script**
   ```bash
   python Multi_agent_script_finale.py
   ```

4. **Outputs**
   - Probability predictions are saved in `predictions.xlsx`
   - Reasoning logs are stored in `logs/`
  
---

## DOI and Citation

This repository is archived on Zenodo for long-term preservation:  
**DOI:** [https://doi.org/10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)

**APA citation example:**
```
Loria, L. (2025). MSc thesis appendix – LLM-Based Multi-Agent Prediction for M&A Value Creation. Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX
```

---

## License

This repository is distributed under the MIT License.  
See `LICENSE` for details.

---

## Notes on Data Privacy

- Full datasets used in the thesis (deal texts, CAR values) cannot be shared due to confidentiality constraints.
- Scripts and logs are provided to ensure transparency and reproducibility of the methodology.

---
 
