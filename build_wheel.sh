#!/bin/bash

if python3 setup.py sdist bdist_wheel ; then
   echo Wheel built.
fi

