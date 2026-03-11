---
description: Python coding rules for ML research projects
globs: ["**/*.py"]
---

Follow these rules when generating or modifying Python code:

1. Code must be clean and readable.
2. Follow PEP8 style guidelines.
3. Use clear variable names related to machine learning tasks.
4. Avoid unnecessary global variables.
5. Add concise comments explaining important logic.
6. Prefer modular functions instead of long scripts.
7. When writing training code, separate:
   - dataset loading
      - model definition
         - training loop
            - evaluation
            8. When modifying existing code, preserve original functionality unless explicitly asked to change it.
            
