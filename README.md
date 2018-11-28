# UserReputation
Predicts the reputation of users on Stack Overflow.

Library Installs
----------------
To install all libraries, you must have Python and pip installed on your machine. If you do not have these, follow [this](https://www.python.org/downloads/) link for instructions to download/install Python and [this](https://pip.pypa.io/en/stable/installing/) link for pip.

To install all the necessary python libraries, run
```bash
./installs.sh
```

Run Code
--------
To **train** and see printed **accuracy** numbers, run
```bash
python train.py
```
This saves our trained model to be later used in test.py. The printed output in the terminal is a verified accuracy of our model after running kforld cross-validation on the dataset given in the assignment.


To **test** our model on new test data, run
```bash
python test.py <path_to_dataset>
```
The predictions will be written to a file called output.txt in the same directory as the input file.
