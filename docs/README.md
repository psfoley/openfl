`
make clean
sphinx-apidoc -f -o . ../tfedlrn
make html
cd _build/html
python -m http.server

`
