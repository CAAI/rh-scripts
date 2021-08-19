# rh-scripts
Private scripts used at CAAI group at Rigshospitalet  

Most scripts are compatible with most python versions.  
Some scripts require python3.8+.

## HOW TO INSTALL

### Using pip
Install only to your user. Go to your virtual environment. Run:
```
git clone https://github.com/CAAI/rh-scripts.git && cd rh-scripts
pip install .
```

### Config.ini
If you need to use the scripts to move data to e.g. DALI / VIA3 servers, you need to set up your config.
Copy the config, e.g. to /opt/caai/share/:
```
cp dicom/config-default.ini /opt/caai/share/config.ini
```
After this, modify the tags needed in the config file, e.g. setting your own AET and add each server you need to transfer data to.
Finally, add to your .bash_profile:
```
# rh-script configuration
export CAAI=/opt/caai/
```
