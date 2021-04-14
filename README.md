# rh-scripts
Private scripts used at CAAI group at Rigshospitalet  

Most scripts are compatible with most python versions.  
Some scripts require python3.8+.

## HOW TO INSTALL

### Using pip
Install only to your user. Go to your virtual environment. Run:
```
git clone https://github.com/CAAI/rh-scripts.git && cd rh-scripts
pip install -e
```

### Using Cmake
To install files system-wide.
```
git clone https://github.com/CAAI/rh-scripts.git && cd rh-scripts
pip3 install -r requirements.txt
mkdir build && cd build
cmake ..
make && make install
```
## POST INSTALLATION
Add "source /opt/caai/toolkit-config.sh" to .bashrc / .bash_profile 

## CHECK INSTALLATION
Perform unittest for python scripts.
```
python -m unittest discover -s /opt/caai/tests -v
```

## KNOWN ISSUES

### Install on ubuntu
Installation does not have access to write to /opt.

#### Solution:
Prior to install run:
```
sudo mkdir /opt/caai
sudo chown -R <user>:<group> /opt/caai
```
Proceed to install with "make install".
