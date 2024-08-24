# Install python venv
python3.10 -m venv .venv

# Activate venv and install pip
source .venv/bin/activate
python3 -m pip install --upgrade pip

# install pip-tools and compile dependencies
python3 -m pip install pip-tools
pip-compile

# Install requirements
python3 -m pip install -r requirements.txt