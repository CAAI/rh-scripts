# rh-scripts
Scripts used at CAAI group at Rigshospitalet

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