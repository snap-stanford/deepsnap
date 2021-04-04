## Documentation Build Instructions

Install the required package for Sphinx first
```sh
pip install -r requirements.txt
```
Then run the following code to generate the documentation
```sh
sphinx-apidoc -f -o source ../deepsnap/ && make html
```
or just
```sh
make html
```
Generated html files are in `build/html` directory. You can open the `index.html`.

You might need to run following command before rebuilding the documentation
```sh
make clean
```