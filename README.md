# Earthquake size estimation

## Description

This repository contains functionality and a Jupyter notebook to explore earthquake size estimation and reproduce
figures use in <reference to manuscript once it's on a pre-print server>

## Configuration

A python environment can be created using conda/mamba from [env.yml](env.yml), i.e.:
```bash
mamba create -f env.yml
mamba activate earthquake-size-estimation
```
If there are problems with the availability of packages, it is worth trying the light enviroment file [env-light.yml](env-light.yml), i.e.:
```bash
mamba create -f env-light.yml
mamba activate earthquake-size-estimation
```

## Usage

See [unbiased_estimate_paper.ipynb](unbiased_estimate_paper.ipynb) for a usage example.

## License

MIT License

Copyright (c) 2025. TNO

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.