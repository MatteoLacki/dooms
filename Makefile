make:
    echo "Welcome to Project 'dooms'"

upload_test_pypi:
    twine check dist/*
    python -m pip install --upgrade twine
    twine upload --repository testpypi dist/*

upload_pypi:
    twine check dist/*
    python -m pip install --upgrade twine
    twine upload dist/* 

ve_dooms:
    python3 -m venv ve_dooms
