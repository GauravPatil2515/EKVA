# Contributing to EKVA v2

Thank you for your interest in contributing to **EKVA v2** (Expert-Conditioned Key-Value Cache Token Retention for Sparse MoE LLMs)! We welcome contributions from systems engineers, machine learning researchers, and open-source enthusiasts.

---

## 🛠️ Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/GauravPatil2515/EKVA.git
   cd EKVA
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies in editable mode:**
   ```bash
   pip install --upgrade pip
   pip install -e ".[dev,models,kernel]"
   ```

4. **Verify the installation:**
   ```bash
   pytest tests/ -v
   ```
   All 31 unit tests should pass.

---

## 🧪 Testing Guidelines

Before opening a pull request, ensure all tests pass:

```bash
pytest tests/ -v --tb=short
```

If you are modifying the retention policies or Triton kernels:
- Ensure exact-sum budget constraint integrity: $\sum_{i=1}^E b_i = B$.
- Verify that retained tokens preserve ascending chronological order to avoid scrambling RoPE positional embeddings.
- If GPU hardware is available, benchmark with:
  ```bash
  python3 experiments/analytical_roofline_model.py
  ```

---

## 📐 Pull Request Workflow

1. Fork the repository and create your feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Commit your changes using conventional commit messages (`feat:`, `fix:`, `docs:`, `perf:`):
   ```bash
   git commit -m "feat(kernel): optimize Triton block tile size for A100 SRAM"
   ```
3. Push to your branch and open a Pull Request against `main`.
4. Provide a clear description of your changes, the motivation, and any benchmark or test results.

---

## 📜 License
By contributing to EKVA, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).

