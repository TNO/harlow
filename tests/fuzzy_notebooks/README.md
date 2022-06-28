This is the start of FLOLA experimentation in Python
=============================================

## Setup
<!-- Create a Python virtual environment and install the requirements -->
- Create a virtual environment
```
python -m venv flola <directory_to_install>
```
If the directory has spaces, use `""` to encapsulate the path.
- Activate the environment
```
.\<directory_installed>\Scripts\activate
```
- Check pip installation (sometimes gives errors)
```
python -m pip install -U pip
```
- Install the requirements
```
python -m pip install -r requirements.txt
```
## Checking the Scikit Fuzzy

the IF part is called the antecedent and the THEN part is the consequent.

## Hyperparameters set-up

- $σ_c$ = 0.3
- $σ_{al}$ = 0.27
- $σ_{ah}$ = 0.3

## Theory
- Point $p_r$
- The set of cross-polytope is $N(p_r)$

How to determine the neighborhood $N(p_r)$  of a reference point $p_r$
- Include all points within a certain range $a$ of the reference point: $N(p_r)=\{p | p \in P_r, ||p-p_r|| <a\}$

- $a$ controls the part of the input space that is included in the gradient estimation ('local' for $p_r$):

    $a = \frac{2}{K} \sum_{j=1}^{K}||n_j - p_r ||$  , $K=4d$

## Cohesion & Adhesion section 4
Definitions $\forall p \in N(p_r)$

$C(p_r)$ and $A(p_r)$ are vectors which represent
cohesion and adhesion values for all neighbors of $p_r$.

- $C(p_r, p) = ||p=p_r ||$
- $A(p_r, p) = min_{q\in P_r} || q-p||$

## Mandani FIS

Consists of a fuzzifier, an inference engine and a defuzzifier

- Fuzzifier: Maps crisp inputs of linguistic variables to fuzzy set memberships, using provided membership functions
- Inference engine: Rule-based that processes rules of the form "if-then"
- Deffuzifier: Use the centroid method

## Fuzzy-based neigbor weight assignment

FIS $S$ is proposed to assign weights to each point in $N(p_r)$.
- Input (2): Cohesion & Adhesion.
- Output (1): Weight

**Points with high cohesion and low adhesion
are preferred and are assigned high weights, while low cohesion and/or high adhesion
result in low weights.**

## Norm and t-conorm

- API file on fuzzymath -> fuzzy_logic.py
