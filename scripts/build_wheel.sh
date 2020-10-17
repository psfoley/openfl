#!/bin/bash

if python3 setup.py sdist bdist_wheel ; then
   echo "Pip wheel built and installed in dist directory"
fi
