set -e

# Summary:
#   1. Creates a virtual environment in LOCAL_VENV_FOLDER.
#   2. Installs dependencies from $LOCAL_GIT_REPO_FOLDER/requirements.txt
#   3. Makes sure helios compiled packages are installed when available.
#      When not avaiable, online pypi packages will be used.
# Pre-requisites:
#   1. Ensure pip has internet access (be on a login node).
#   2. Clone this git repository to your helios home folder.
#   3. Edit LOCAL_GIT_REPO_FOLDER and LOCAL_VENV_FOLDER as necessary.
# Example usage:
#   /create_venv.sh

LOCAL_GIT_REPO_FOLDER=~/ift6759-project1
LOCAL_VENV_FOLDER=~/ift6759-project1-venv

if [ -d "$LOCAL_VENV_FOLDER" ];
then
  echo "Error: The folder $LOCAL_VENV_FOLDER already exists. Exiting."
  exit 1
fi

module load hdf5/1.10.3
module load python/3.7.4
virtualenv --no-download $LOCAL_VENV_FOLDER
source $LOCAL_VENV_FOLDER/bin/activate

avail_helios_packages=$(ls /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/generic)
required_packages=$(sed -e 's/#.*$//' -e '/^$/d' $LOCAL_GIT_REPO_FOLDER/requirements.txt)
if [[ "$required_packages" == *+(<|>)* ]]
then
  echo "Error: One of the requirements uses < or >. This is not supported. Full list of requirements: $required_packages. Exiting."
  exit 1
fi

online_packages=""
offline_packages=""
for requirement in $required_packages; do
  requirement_split=( ${requirement//[=]/ } )
  package_name="${requirement_split[0]}"
  version="${requirement_split[1]}"
  if [[ -z $version ]]
  then
    version="[0-9]+.[0-9]+.[0-9]+"
  fi
  if [[ -z $(echo $avail_helios_packages | egrep $package_name-$version) ]]
  then
    online_packages+="$requirement "
  else
    offline_packages+="$requirement "
  fi
done

echo "Upgrading pip..."
pip install --no-index --upgrade pip
echo "Installing required packages available on helios: $offline_packages..."
pip install --no-index $offline_packages
echo "Installing required packages not available on helios: $online_packages..."
pip install $online_packages
