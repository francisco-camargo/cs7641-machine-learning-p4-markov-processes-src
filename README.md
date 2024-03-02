# CS7641 Machine Learning - Project 4: Markov Decision Processes - Francisco Camargo

## Assignment Instructions

The instructions for this assignment can be found within `p4 - Markov Decision Processes.pdf`

## Installing environment

Download and install Python 3.11.0 from https://www.python.org/downloads/

To update `pip`, use

`python -m pip install --upgrade pip`

To create an environment via the terminal, use

`python -m venv env`

To activate environment, use

`env/Scripts/activate`

To install libraries, use

`pip install -r requirements.txt`

To deactivate an active environment, use

`deactivate`

## Running the Code

Open `main.py`, from the `experiment_list` list uncomment which ever experiments you want to run. Then run `python main.py`.

## LaTeX template

The LaTeX template I used for this report comes from the Vision Stanford [website](http://vision.stanford.edu/cs598_spring07/report_templates/)

# Change to how Frozen Lake rewards are defined

In gymnasium.envs.toy_text.frozen_lake.py I changed the lines from

```Python
def update_probability_matrix(row, col, action):
    newrow, newcol = inc(row, col, action)
    newstate = to_s(newrow, newcol)
    newletter = desc[newrow, newcol]
    terminated = bytes(newletter) in b"GH"
    return newstate, reward, terminated
```

to now read

```Python
def update_probability_matrix(row, col, action):
    newrow, newcol = inc(row, col, action)
    newstate = to_s(newrow, newcol)
    newletter = desc[newrow, newcol]
    terminated = bytes(newletter) in b"GH"
    if newletter == b'G': reward = 1.0
    elif newletter == b'H': reward = -10.0
    else: reward = -0.000001
    return newstate, reward, terminated
```

# Change to bettermdptools

In `algorithms.planner.py`, make the following bug fix:

Change `while i < n_iters` to `while i < n_iters-1` in the `policy_iteration()` function
