# Curiosity-Driven Zero-Shot Object Navigation with Vision-Language Models

## Abstract
Zero-shot object navigation (ZSON) in unseen environments poses a significant challenge due to the absence of object-specific priors and the need for efficient exploration. Existing approaches often struggle with ineffective search strategies and repeated visits to irrelevant areas.
In this paper, we introduce a curiosity-driven framework that leverages the commonsense reasoning capabilities of vision-language models (VLMs) to guide exploration. At each step, the agent estimates the semantic plausibility of regions based on language-conditioned visual cues, constructing a dynamic value map that promotes informative regions and suppresses redundancy.
The core contribution of this work is integrating VLM-based scene understanding into the curiosity mechanism, enabling the agent to make human-like judgments about environmental relevance during navigation.
Extensive experiments on the HM3D benchmark show that our method achieves a 12.1% absolute improvement in Success Rate (SR) over strong baselines (from 56.5% to 68.6%). Qualitative analysis further confirms that the proposed strategy leads to more efficient and goal-directed exploration.

## Installation

Set up the conda environment (Linux, Python 3.9)

Install the challenge-2022 version of [Habitat-Sim](https://github.com/facebookresearch/habitat-sim) (headless with no Bullet physics) with:

```bash
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim; git checkout tags/challenge-2022; 
pip install -r requirements.txt; 
python setup.py install --headless
python setup.py install # (for Mac OS)
```

<!-- Follow instructions at [https://github.com/TRI-ML/prismatic-vlms](https://github.com/TRI-ML/prismatic-vlms) to set up PrismaticVLM. -->
Set up [Prismatic VLM](https://github.com/TRI-ML/prismatic-vlms) with the submodule:

```bash
cd prismatic-vlms && pip install -e .
```

## Usage

Run our method in Habitat-Sim:

```bash
python run_exp.py -cf cfg/vlm_exp.yaml
```

## Acknowledgement
Our work is based on [explore-eqa](https://github.com/Stanford-ILIAD/explore-eqa).

