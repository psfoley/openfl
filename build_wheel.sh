#!/bin/bash

if python setup.py sdist bdist_wheel ; then
   echo Wheel built.
fi

