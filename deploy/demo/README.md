# Public demo data

`model.joblib` is the project’s LightGBM 2.0.0 artifact trained on PaySim. Reproduce it with the full-data training command in the project README, then copy `artifacts/phase3/model.joblib` here. The small transaction sample includes deliberately selected fraud examples, so its rates describe the demo only.

The sample is adapted from Edgar Lopez-Rojas’s [PaySim dataset](https://www.kaggle.com/datasets/ealaxi/paysim1/metadata), published under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Changes include selecting a subset of held-out events and assigning demo transaction IDs. The adapted CSV is distributed under the same license.

`scores.json` adds the model outputs and SHAP contributions for those same transactions and is distributed under CC BY-SA 4.0.
