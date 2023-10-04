## Data mining project @UQ


TODO:
- [x] Data preprocessing   
- [ ] Making another way of detecting anomalies for the class 'car'. Given that it doesn't follow a normal distribution. 
- [ ] Creating k-NN classifier with k-fold CV.
- [ ] Creating k-means classifier with k-fold CV.
- [ ] Creating decision tree classifier with k-fold CV.
- [ ] Creating random forest classifier with k-fold CV.
  
---

# My Python Project

This is a brief description of your Python project. Explain what the project does, who it's for, and why you created it.

## Prerequisites

- **Python**: This project requires Python 3.7+ (or the version you used).
- **Poetry**: This project uses Poetry for dependency management.

## Getting Started

1. **Install Poetry (if you haven't already)**:

   Detailed instructions for various operating systems can be found at [Poetry's installation documentation](https://python-poetry.org/docs/#installation). Here's a quick start for common OS:

   - On macOS and Linux:

     ```bash
     curl -sSL https://install.python-poetry.org | bash
     ```

   - On Windows (using PowerShell):

     ```bash
     (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
     ```

2. **Clone this repository**:

   ```bash
   git clone https://github.com/yourusername/yourprojectname.git
   cd yourprojectname
   ```

3. **Install dependencies**:

   ```bash
   poetry install
   ```

## Usage

1. **Activate the virtual environment**:

   ```bash
   poetry shell
   ```

2. **Run the main script or application**:

   ```bash
   python main.py
   ```