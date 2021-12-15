# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- Trained by: Ben E
- Date: 14/12/21
- Version: 3
- Algorithm: Random Forest classifier
- Paper: NA
- License: NA

## Intended Use
- Uses: Predicting whether individual earns below or above $50k
- Users: Marketing targeting
- Out of scope: NA

## Training Data
- Categorical Features:
  - workclass
  - education
  - marital-status
  - occupation
  - relationship
  - race
  - sex
  - native-country
- Numerical Features:
  - fnlgt
  - age
  - capital-gain
  - capital-loss
  - hours-per-week
  - education-num
- target
  - salary

## Evaluation Data
- Hold-out of 20%

## Metrics
- Precision: 0.72
- Recall: 0.64
- F1 Score: 0.68

## Ethical Considerations
- Based on a sample at a single point in time.

## Caveats and Recommendations
- Small dataset of 32k instances.