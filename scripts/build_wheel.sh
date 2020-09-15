#!/bin/bash

if python setup.py sdist bdist_wheel ; then
   echo "Pip wheel built and installed in dist directory"
fi
