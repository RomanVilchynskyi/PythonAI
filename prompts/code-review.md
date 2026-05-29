# Code Review Skill

When reviewing code:

* Detect bugs, security vulnerabilities, and code-smells
* Detect possible null reference issues
* Suggest better naming for variables and methods
* Check SOLID principles and clean architecture violations
* Detect duplicated logic
* Suggest performance improvements
* Explain issues briefly and clearly
* Prefer clean and maintainable code
* **Propose creating a new file named `code-review-report.md`** containing the review results.
* **Always output the final result strictly as a Markdown table** inside this file, without any conversational filler before or after it.

Use the following table template for the file content:

| Category (Bug / Vulnerability / Code Smell) | Line or Method | Issue Description | Severity (Low / Medium / High / Critical) | Recommendation | Fixed Code Example |
|---|---|---|---|---|---|
| Category name | Method() | Brief explanation | High | What to do | `fixed code snippet` |