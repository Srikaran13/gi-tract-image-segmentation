# 6.8300 - Advances in Computer Vision Course Project

This course project is adapted from the Kaggle competition found here: [UW-Madison GI Tract Image Segmentation](https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation). This paper introduces an innovative approach to
computer-aided diagnosis for radiation therapy planning
in gastrointestinal (GI) cancer patients. By applying deep
learning methodologies on the segmentation of stomach and
intestines from MRI scans, we aim to expedite treatment
procedures and optimize therapy outcomes. By leveraging
anonymized MRIs sourced from the UW-Madison Carbone
Cancer Center, we employ advanced models such as UNet,
DeepLabv3+, and R-CNNs, exploring multi-task learning
through supervised approaches. We found that Efficient Net
excels at segmenting the stomach and large bowels, while
Mask R-CNN delivers the most balanced high performance
across all three organs. This innovative approach has the
potential to revolutionize radiation therapy planning, expediting treatment procedures and enabling radiation oncologists to concentrate more on treatment optimization, ultimately enhancing the overall effectiveness of therapies.

## Project Report
You can view the detailed project report [here](./Project%20Report.pdf).

## Repository Structure
### Notebooks
- **Processing**: All the Jupyter notebooks used for processing the data can be found in the `notebooks` folder.

### Source Code 
- **Models**: Implementation of various models can be found in `model.py`.
- **Data Handling**: Scripts for data preprocessing and loading are in `data.py`.
- **Loss Functions**: Custom loss functions used in the project are defined in `loss.py`.


Please follow the steps below to set up and run the project.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

What things you need to install the software and how to install them:

```bash
pip install -r requirements.txt
```

### Installation
A step-by-step series of examples that tell you how to get a development environment running:

## Dataset Setup

Follow these steps to download and prepare the dataset for the project:

### Download the Dataset

1. **Visit the Kaggle Competition Page**:
   - Navigate to the [UW-Madison GI Tract Image Segmentation competition page on Kaggle](https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation).

2. **Download the Dataset**:
   - Click on the "Download All" button to download the dataset zip file.

3. **Place the Dataset in the Project Folder**:
   - Extract the contents of the downloaded zip file.
   - Move the extracted folder to your project directory.

### Prepare the Data

1. **Rename the Extracted Folder**:
   - Rename the extracted dataset folder to `datasets` to maintain consistency with project scripts.

By following these steps, you will have the required data ready in the appropriate format and location for further processing and analysis.



