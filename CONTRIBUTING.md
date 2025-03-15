# Contributing to EvoAgentX

Thank you for considering contributing to **EvoAgentX** – an automatic agentic workflow generation and evolving framework! We appreciate your interest and contributions to improving the project.

## 🚀 Getting Started

1. **Fork the Repository**: Click the 'Fork' button on the top right of this repo to create your own copy.
2. **Clone Your Fork**:
   ```bash
   git clone https://github.com/your-username/EvoAgentX.git
   ```
3. **Navigate to the Project Directory**:
   ```bash
   cd EvoAgentX
   ```
4. **Create a Virtual Environment**:
   ```bash
   conda create -n agent python=3.10
   conda activate agent
   ```
5. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🛠 How to Contribute

### **1. Reporting Bugs** 🐞
- Ensure the bug has not been reported in the [Issues](https://github.com/EvoAgentX/issues) tab.
- Provide detailed steps to reproduce the issue.
- Include logs, screenshots, and environment details.
- Use a clear title and concise description.

### **2. Suggesting Features** 💡
- Check if a similar feature request exists in [Issues](https://github.com/EvoAgentX/issues).
- Provide a use case explaining why this feature is beneficial.
- Suggest a possible implementation approach if possible.

### **3. Code Contributions** 👨‍💻
#### **Step 1: Create a Branch**
Before making changes, create a new branch:
```bash
git checkout -b feature/your-feature-name
```

#### **Step 2: Implement Your Changes**
- Follow the project’s coding style.
- Write clean, maintainable, and well-documented code.
- Add relevant unit tests for your changes.

#### **Step 3: Commit Your Changes**
- Follow conventional commit messages:
  ```bash
  git commit -m "feat: add new agent workflow module"
  ```
- Use descriptive commit messages.

#### **Step 4: Push Your Changes**
```bash
git push origin feature/your-feature-name
```

#### **Step 5: Submit a Pull Request**
- Navigate to the main repository and create a **Pull Request** (PR).
- Provide a clear description of your changes.
- Reference the related issue if applicable.
- Wait for a review and respond to feedback promptly.

## 📏 Coding Guidelines

### Google-Style Docstring Guidelines

All Python functions, classes, and modules in this project should use **Google-style docstrings**. Below are the guidelines and examples that contributors must follow to maintain consistency and clarity throughout the codebase.

---


#### Basic Structure

A typical Google-style docstring contains:

1. **Short Summary**  
   A brief statement of what the function or class does.
2. **Args**  
   A list of parameters with their types (optional) and descriptions.
3. **Returns** or **Yields**  
   Explains what the function returns or yields, along with types and descriptions.
4. **Raises** (optional)  
   Documents any exceptions that might be raised.
5. **Other sections** (optional)  
   Such as **Example(s)**, **Attributes** (for classes), etc.

**Example Layout**:

```python
class Calculator:
    """A simple calculator class to demonstrate Google-style docstrings.

    This class provides basic arithmetic operations. Advanced features can be
    added as needed.

    Attributes:
        last_result (float): Stores the result of the most recent operation.
    """

    def __init__(self):
        """Initializes the Calculator with a default result of 0."""
        self.last_result = 0

    def add(self, a, b):
        """Adds two numbers and updates `last_result`.

        Args:
            a (float): The first number.
            b (float): The second number.

        Returns:
            float: The sum of `a` and `b`.
        """
        self.last_result = a + b
        return self.last_result

    def subtract(self, a, b):
        """Subtracts b from a and updates `last_result`.

        Args:
            a (float): The number to be subtracted from.
            b (float): The number to subtract.

        Returns:
            float: The result of `a - b`.
        """
        self.last_result = a - b
        return self.last_result
```

## 🧪 Testing
Before submitting a PR, ensure that all tests pass:
```bash
pytest tests/
```

## 🔄 Syncing Your Fork
To stay updated with the latest changes:
```bash
git fetch upstream
```
```bash
git merge upstream/main
```

## 🤝 Community Guidelines
- Be respectful and inclusive.
- Provide constructive feedback.
- Keep discussions relevant to the project.

## 📩 Contact
For any questions, feel free to open an issue or reach out to the maintainers.

Happy coding! 🚀
