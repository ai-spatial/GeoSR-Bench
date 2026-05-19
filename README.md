# GeoSR-Bench


Dataset and code for the paper:


**Beyond Visual Fidelity: Benchmarking Super-Resolution Models for Large-Scale Remote Sensing Imagery via Downstream Task Integration** [[arXiv]](https://arxiv.org/abs/2605.00310)


GeoSR-Bench is a large-scale benchmark for evaluating remote sensing super-resolution models beyond conventional image fidelity metrics, such as PSNR and SSIM. GeoSR-Bench evaluates whether super-resolved images improve downstream Earth observation tasks.

## 🚧 Dataset Upload in Progress

The GeoSR-Bench dataset is currently being uploaded and organized on Hugging Face:

https://huggingface.co/datasets/ai-spatial/GeoSR-Bench

Some files, metadata, subsets, and directory structures may still be incomplete or subject to change during the upload process. An official release announcement will be made once the upload and verification process is complete.

## Benchmark Tasks

GeoSR-Bench covers two cross-platform super-resolution settings:

- **MODIS → Landsat-8**
- **Sentinel-2 → NAIP**

It also includes multiple downstream task datasets for evaluating the practical utility of super-resolution outputs, such as:

- Land cover classification
- Road and building mapping
- Crop mapping
- Water mapping
- Gross primary productivity estimation
- Canopy height estimation

## Repository Status

The trained models will be released soon.
