# Install python venv
python3.10 -m venv .venv

# Activate venv and install pip
source .venv/bin/activate
python3 -m pip install --upgrade pip

# Install requirements
python3 -m pip install -r requirements.txt