# FedWKD: Federated Learning Weighted Aggregation with Knowledge Distillation For IoT Forecasting


**Author:** Bouchra Fakher  
**Published in:** Elsevier's IoT Journal  
**Date:** 2025  
**DOI:** https://doi.org/10.1016/j.iot.2025.101849

## Abstract

Federated Learning (FL) has emerged as a promising solution for decentralized Machine Learning (ML) for heterogeneous, non-independent and identically distributed (non-IID) time-series data. However, the traditional FL methods are prone to overfitting at the client level, which may cause privacy leakage and complicate deployment in highly sensitive data environments. In this paper, we propose a novel approach that integrates Knowledge Distillation (KD) by using distilled information of each client, named logits, alongside global distilled logits to improve local model training while also assigning importance weights for each client during aggregation of received logits and models on the server-side. The proposed method employs KD regularization techniques that align local objectives with global objectives while refraining the server (teacher) from training on a dataset on the server-side by only aggregating the received logits and models using weighted aggregation. Thus, we avoid training overhead and data leakage on the server and use weighting based on logits for each training round, achieving better convergence for non-IID data. Experimental results highlight its ability to improve forecasting metrics compared to other methods such as CADIS and FEDGKD, using loss, error, and execution time metrics, hence bettering generalization and minimizing client drift and bias.

# If you use this work, please cite it as follows:

```bibtex
@article{fakher2025fedwkd,
  title={FedWKD: Federated Learning Weighted Aggregation with Knowledge Distillation for IoT Forecasting},
  author={Fakher, Bouchra and Brahmia, Mohamed El Amine and Bennis, Ismail and Abouaissa, Abdelhafid},
  journal={Internet of Things},
  pages={101849},
  year={2025},
  publisher={Elsevier}
}
