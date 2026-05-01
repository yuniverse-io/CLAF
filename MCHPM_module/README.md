# MCHPM

Official implementation of:
> Lim, H., Park, S., Li, Q., Li, X., & Kim, J. (2026).
**What makes a review helpful? A multimodal prediction model in e-commerce**. 
_Electronic Commerce Research and Applications_, 76, 101586. [Paper](https://doi.org/10.1016/j.elerap.2026.101586)

## Overview
This repository is the official implementation of MCHPM (Multimodal Cue-based Helpfulness Prediction Model), published in *Electronic Commerce Research and Applications* (2026).

Most multimodal review helpfulness prediction (MRHP) models rely on deep semantic representations of text and images and overlook surface-level cues such as readability, sentiment intensity, and image quality. MCHPM addresses this gap by drawing on the **Elaboration Likelihood Model (ELM)** from consumer psychology, which describes how readers process information through two parallel routes — a *central* route based on careful cognitive engagement, and a *peripheral* route based on superficial heuristics.

For each modality (text and image), MCHPM extracts both **central cues** (deep semantic representations from BERT and VGG-16) and **peripheral cues** (surface-level features like readability and image clarity). Within each modality, central and peripheral cues are integrated through co-attention; the resulting text and image representations are then fused via a Gated Multimodal Unit (GMU) that adaptively weights the two modalities.

The model predicts a continuous review-helpfulness score, defined as `log(1 + helpful_vote)`, as a regression target. Quantitative comparisons against unimodal and multimodal baselines on large-scale Amazon datasets are reported in [Experimental Results](#experimental-results).

## Repository Structure

```bash
├── data/
│   ├── raw/                        # Source datasets — place {fname}.jsonl.gz here
│   ├── processed/                  # Pipeline parquet caches (labeled / cued)
│   ├── review_images/              # Downloaded review images, grouped by dataset name
│   └── mchpm_architecture.png
│
├── model/
│   ├── mchpm.py                    # MCHPM architecture, trainer, and tester
│   └── save/                       # Best checkpoint per dataset (best.pth)
│
├── src/
│   ├── config.yaml                 # Single source of truth for all hyperparameters
│   ├── data_processing.py          # DataProcessor pipeline + DataLoader + peripheral standardizer
│   ├── text_cue_extractor.py       # BERT central + TextBlob/textstat peripheral cues
│   ├── image_cue_extractor.py      # VGG-16 central + OpenCV peripheral cues
│   ├── review_image_downloader.py  # Parallel review image downloader (cache-aware)
│   ├── text_processing.py          # Review text cleaning + English filter
│   ├── path.py                     # Project path constants (auto-creates runtime folders)
│   └── utils.py                    # Metrics, parquet/yaml/seed helpers
│
├── main.py                         # Entry point: data preparation → train → test
├── requirements.txt
├── README.md
└── .gitignore
```

## Model Description

MCHPM consists of three sequential modules. The full architecture is illustrated below.

<p align="center">
  <img src="data/mchpm_architecture.png" alt="MCHPM Architecture" width="800">
</p>

### 1. Multi-Cue Extraction Module
Extracts central and peripheral cues from review text and images in parallel.

**Central cues** (deep semantic representations):
- Text: BERT `[CLS]` embedding
- Image: VGG-16 `fc2` activation

**Peripheral cues** (surface-level features):
- Text — polarity, subjectivity, readability, extremity
- Image — brightness, contrast, saturation, edge intensity

Implementation: [`src/text_cue_extractor.py`](src/text_cue_extractor.py), [`src/image_cue_extractor.py`](src/image_cue_extractor.py).

### 2. Cue-Integration Module
Within each modality, central and peripheral representations attend to each other through co-attention: central queries peripheral, peripheral queries central, and the two attended outputs are combined via element-wise multiplication. The same pattern is applied independently to the text and image sides, yielding modality-specific integrated vectors `O_t` and `O_v`.

Implementation: `CoAttentionBlock` in [`model/mchpm.py`](model/mchpm.py).

### 3. Multimodal Fusion Module
The integrated text and image vectors are passed through `tanh` projections, then fused by a Gated Multimodal Unit. A sigmoid gate `z`, computed from the concatenated representations, adaptively weights the contribution of each modality. The fused vector is forwarded to an MLP regressor that outputs the predicted helpfulness score.

Implementation: `MCHPM.gate_layer` and `MCHPM.regressor` in [`model/mchpm.py`](model/mchpm.py).

## How to Run

### Configuration
All hyperparameters live in [`src/config.yaml`](src/config.yaml) — it is the single source of truth. Defaults reproduce the paper experiments. Edit values (dataset name, batch size, learning rate, model dims, etc.) before running if you want to override.

The `torch==2.3.1+cu121` / `torchvision==0.18.1+cu121` wheels pinned in [`requirements.txt`](requirements.txt) target an RTX 3080 Ti (CUDA 12.1, Ampere); for other GPUs or a CPU-only setup, follow the header comment in `requirements.txt`.

End-to-end run from a fresh checkout:
```bash
conda create -n mchpm python=3.11
conda activate mchpm
pip install -r requirements.txt
python main.py
```

### Data Preparation
Place the dataset as `data/raw/{fname}.jsonl.gz` where `{fname}` matches `data.fname` in `config.yaml`.

**Required columns in raw JSONL**:
`user_id`, `parent_asin`, `timestamp`, `text`, `images`, `helpful_vote`, `verified_purchase`, `title`
(`text`, `images`, `title` are renamed to `raw_review`, `review_images`, `review_title`)

Pipeline writes two cached parquets under `data/processed/`. The train/test shuffle split runs in memory on every run (no on-disk cache). Each stage requires the listed columns:

**`{fname}_labeled.parquet`** — after row filters, text cleaning, and label construction:
`user_id`, `parent_asin`, `timestamp`, `review_date`, `raw_review`, `clean_review`, `review_images`, `review_title`, `helpful_vote`, `label`

**`{fname}_cued.parquet`** — after image download and cue extraction:
labeled columns + `review_image_paths`, `review_text_central`, `review_text_peripheral`, `review_image_central`, `review_image_peripheral`

To reuse externally-extracted BERT/VGG features, save the data as `{fname}_labeled.parquet` with `review_text_central` and/or `review_image_central` columns pre-populated. The pipeline will skip BERT/VGG and only compute peripheral cues.

### Re-runs and caching
On every call to `python main.py`, the pipeline auto-skips any cache layer already on disk (cued → labeled → image folder), so subsequent runs reuse prior work. The train/test split is rebuilt fresh in memory each run, so changes to `test_size`, `random_state`, or `val_ratio` take effect immediately on the next run. To force an upstream stage to re-run, delete the corresponding parquet (or the `data/review_images/{fname}/` folder for image re-downloads).

## Experimental Results

MCHPM was evaluated on two large-scale Amazon review datasets: Cell Phones & Accessories and Electronics.
The results demonstrate that MCHPM consistently outperforms strong unimodal and multimodal baselines across all evaluation metrics, achieving average improvements of 3.864% in MAE, 4.061% in MSE, 2.172% in RMSE, and 6.349% in MAPE compared with the strongest benchmark model.

<div align="center">
  <table> 
    <thead> 
      <tr>
        <th rowspan="2">Model</th>
        <th colspan="4">Cell Phones & Accessories</th> 
        <th colspan="4">Electronics</th> 
      </tr>
      <tr> 
        <th>MAE</th> 
        <th>MSE</th> 
        <th>RMSE</th> 
        <th>MAPE</th>
        <th>MAE</th>
        <th>MSE</th>
        <th>RMSE</th>
        <th>MAPE</th>
      </tr>
    </thead> 
    <tbody> 
      <tr> 
        <td>LSTM</td> 
        <td>0.647</td><td>0.821</td><td>0.849</td><td>56.702</td> 
        <td>0.711</td><td>0.896</td><td>0.946</td><td>57.678</td> 
      </tr>
      <tr>
        <td>TNN</td>
        <td>0.643</td><td>0.714</td><td>0.845</td><td>56.650</td>
        <td>0.722</td><td>0.904</td><td>0.851</td><td>59.556</td>
      </tr>
      <tr>
        <td>DMAF</td>
        <td>0.625</td><td>0.691</td><td>0.836</td><td>53.139</td>
        <td>0.697</td><td>0.880</td><td>0.939</td><td>55.198</td>
      </tr>
      <tr>
        <td>CS-IMD</td>
        <td>0.615</td><td>0.681</td><td>0.825</td><td>52.392</td>
        <td>0.687</td><td>0.831</td><td>0.912</td><td>56.032</td>
      </tr>
      <tr>
        <td><b>MFRHP (Proposed)</b></td>
        <td><b>0.625</b></td><td><b>0.695</b></td><td><b>0.837</b></td><td><b>53.116</b></td>
        <td><b>0.695</b></td><td><b>0.840</b></td><td><b>0.916</b></td><td><b>57.488</b></td>
      </tr>
    </tbody>
  </table>
</div>
      
## Citation

If you use this repository in your research, please cite:

```bibtex
@article{LIM2026101586,
  title = {What makes a review helpful? A multimodal prediction model in e-commerce},
  author = {Heena Lim and Seonu Park and Qinglong Li and Xinzhe Li and Jaekyeong Kim},
  journal = {Electronic Commerce Research and Applications},
  volume = {76},
  pages = {101586},
  year = {2026},
  doi = {10.1016/j.elerap.2026.101586}  
}
```

## Contact

For research inquiries or collaborations, please contact:  

**Seonu Park**  
Ph.D. Student, Department of Big Data Analytics  
Kyung Hee University  
Email: sunu0087@khu.ac.kr

**Qinglong Li**  
Assistant Professor, Division of Computer Engineering  
Hansung University  
Email: leecy@hansung.ac.kr

_Last updated: March 2026_
