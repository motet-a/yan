#!/bin/sh -ex

pep8 *.py
python3 -m unittest test
./test_on_real_projects.sh
