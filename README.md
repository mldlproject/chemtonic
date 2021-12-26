# ChemTONIC: Simplifying chemical data curation
![alt text](https://github.com/mldlproject/chemtonic/blob/main/chemtonic.svg)

Data curation is undoubtedly the most essential stage before running any modeling experiments. Now, although commercial and non-commercial tools and software for data curation are available, to our knowledge there are neither tools nor software developed to support entire substages in data curation. Rdkit is a powerful Python library developed to deal with most chemical data issues but it is an enormous source of functions that may confuse newcomers. Besides, since Rdkit provides users with a bunch of functions tackling numerous processing issues only, users need to design their data curation process using Rdkit's supported modules on their own. In other words, newcomers or even experienced users may get into trouble when performing data curation regardless of being supported by Rdkit. 

Two major issues that are commonly seen are lacking (i) programming skills and (ii) appropriate procedures. To solve these two issues, we proposed **ChemTONIC**, a Python library designed for chemical data curation based on Rdkit. With **ChemTONIC**, we provide users with

- A chemical data curation module with some key functions only.

- A standard data curation procedure proposed by [Fourches et al. (2010)](https://pubs.acs.org/doi/10.1021/ci100176x) whose major substages are represented by functions

We hope that **ChemTONIC** can simplify your data curation stage and contribute to accelerating your chemoinformatics research.
