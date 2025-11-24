# Research Paper: AI and ML in Object Detection - Trends, Applications, and Future Directions

This repository contains the supplementary materials for the research paper "AI and ML in Object Detection: Trends, Applications, and Future Directions" (published on Zenodo, DOI: [10.5281/zenodo.17672370](https://doi.org/10.5281/zenodo.17672370)). The paper surveys state-of-the-art object detection models (e.g., YOLO, SSD, Faster R-CNN), evaluates them empirically, and introduces a novel Deployability Score for edge deployment.

The code and data here enable reproducibility of the experimental evaluation (Section IV), including metric computation and visualization.

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/Nayan-Bebale/Research_Paper.git
   cd Research_Paper
   ```
2. Install dependencies (Python 3.10+):
   ```
   pip install pandas numpy matplotlib
   ```

## Usage
- **Run main.py**: Processes the raw data, computes averages, and generates visualizations (e.g., accuracy vs. latency plot).
  ```
  python main.py
  ```
- **Run deployability_score.py**: Computes the Deployability Score for models using the formula from the paper.
  ```
  python deployability_score.py
  ```
- **Data**: Use `model_results_final.xlsx` as input—contains raw metrics (accuracy, latency, energy, etc.) from experiments on COCO/ImageNet.

Output: CSV summaries, PNG plots (e.g., fig3_scatter.png), and console logs for verification.

## Files
- **main.py**: Main script for data processing, averaging, and plotting (e.g., Fig. 3 in paper).
- **deployability_score.py**: Script to calculate the Deployability Score = (accuracy × 10) / (latency + energy/100).
- **model_results_final.xlsx**: Raw experimental data (15 models, metrics like mAP, latency, energy).

## Citation
If you use this paper, code, or data, please cite:
```
Bebale, N. (2025). ML & AI in Object Detection Research: Evaluation, Deployability and Trends (2025). Zenodo. https://doi.org/10.5281/zenodo.17672370
```

## License
This repository is licensed under the MIT License. See [LICENSE](LICENSE) for details.

For questions, contact: nayanbebale2003@gmail.com
```
