# only used for local test
# do not run it in production environment

pip uninstall pDeepXL
rm -rf build/ dist/ pDeepXL.egg-info/
python3 setup.py sdist bdist_wheel
pip install .