# Highz-EXP
This is a preliminary version for analyzing data collected from the HighZ-EXP experiment. It includes modules for loading files, processing reflection data, plotting spectra, and converting units.

## Installation

### From GitHub (for Google Colab or local installation)

```python
!pip install git+https://github.com/3aboooody56/Highz-EXP.git
```
For editable install, do 
```python
!pip install -e git+https://github.com/3aboooody56/Highz-EXP.git
```


### For development (editable install)

```bash
git clone https://github.com/3aboooody56/Highz-EXP.git
cd Highz-EXP
pip install -e .
```

## Usage

```python
# Import the package (note: use underscore, not hyphen)
import highz_exp

# Or import specific modules
from highz_exp import file_load, reflection_proc, spec_plot, unit_convert
```