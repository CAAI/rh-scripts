# rh-scripts
Private scripts used at CAAI group at Rigshospitalet  

Most scripts are compatible with most python versions.  
Some scripts require python3.8+.

## HOW TO INSTALL
```
pip3 install -r requirements.txt
mkdir build
cd build
cmake ..
make install
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
sudo make install gives error on install_manifest.txt

#### Solution:
Prior to install run:
```
sudo mkdir /opt/caai
sudo chown -R <user>:<group> /opt/caai
```
Proceed to install with "make install" without sudo.
