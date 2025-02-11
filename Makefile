# Define the virtual environment directory
VENV_DIR = venv

# Create a virtual environment
venv:
	python3 -m venv $(VENV_DIR)

# Activate the virtual environment and install dependencies
install: venv
	$(VENV_DIR)/bin/pip install -r requirements.txt

# Run the main script
run: install
	$(VENV_DIR)/bin/python main.py

# Clean up the virtual environment
clean:
	rm -rf $(VENV_DIR)

# Default target
all: run
