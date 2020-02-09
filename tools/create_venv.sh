# Summary:
#   1. Creates a virtual environment in the current working directory.
#   2. Installs dependencies from $LOCAL_GIT_REPO_FOLDER/requirements.txt
#   3. Makes sure helios compiled packages are installed when available.
#      When not avaiable, online pypi packages will be used.
# Pre-requisites:
#   1. Ensure pip has internet access (be on a login node).
#   2. Clone this git repository to your helios home folder.
#   3. Edit LOCAL_GIT_REPO_FOLDER and LOCAL_VENV_FOLDER as necessary.
# Example usage:
#   ~/ift6759-project1/tools/create_venv.sh

LOCAL_GIT_REPO_FOLDER=~/ift6759-project1
VENV_FOLDER_TO_CREATE=ift6759-project1-venv

module load python/3.7.4
virtualenv --no-download $VENV_FOLDER_TO_CREATE
source $VENV_FOLDER_TO_CREATE/bin/activate

avail_helios_packages=$(ls -lR /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/)
required_packages=$(sed -e 's/#.*$//' -e '/^$/d' $LOCAL_GIT_REPO_FOLDER/requirements.txt)
online_packages=""
offline_packages=""
for requirement in $required_packages; do
  helios_name=$(echo "$requirement" | sed -r 's/[=<>]+/-/g')
  if [[ "$avail_helios_packages" == *"$helios_name"* ]]
  then
    offline_packages+="$requirement "
  else
    online_packages+="$requirement "
  fi
done

echo "Upgrading pip..."
pip install --no-index --upgrade pip
echo "Installing required packages available on helios: $offline_packages ..."
pip install --no-index $offline_packages
echo "Installing required packages not available on helios: $online_packages ..."
pip install $online_packages
