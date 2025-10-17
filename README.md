# Unsupervised Representation Learning + Core Vision Checks  

In this assignment, you’ll explore **unsupervised (and self-supervised) learning** to build visual representations from raw robotics data.  The goal is to see how useful these representations are when you have very limited labeled data.  

---

## Objective  

- Pretrain visual representations on **unlabeled data**.  
- Fine-tune on a **tiny labeled dataset** and compare against training from scratch.  
- Visualize embeddings to see whether the representations make sense.  

---

## What to Do  

### 1. Dataset & Pretraining  
- Use ≤500 **unlabeled frames** from one of:  
  - [TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download)  
  - [KITTI Raw](https://www.cvlibs.net/datasets/kitti/raw_data.php?type=person)  
- Learn representations using one of:  
  - **SimCLR / BYOL / MoCo** (contrastive/self-supervised methods)  
  - OR **k-means** on embeddings  

### 2. Small-Data Testing  
- Take ≤500 **labeled samples**.  
- Fine-tune your pretrained representations.  
- Compare with a model trained **from scratch**.  

### 3. Visualization  
- Project embeddings with **t-SNE** or **PCA**.  
- Show clustering quality visually.  

---

## Optional (Extra Credit)  

If you want to go further:  
- Compare **classical feature descriptors** (SIFT, ORB) vs CNN embeddings.  
- Show both **t-SNE vs PCA** and explain intuitively.  
- Compute clustering metrics (silhouette score, Davies–Bouldin index).  
- Run a **data augmentation study** in contrastive learning. 

---

## Submission and deadline
- Submit your work by committing your code to this repository within 4 days of accepting the assignment.
- Submissions made to personal repositories will not be reviewed; ensure all work is pushed to the designated repository provided for you.
---

## 💡 Notes  

- Keep labeled data usage small (≤500 samples).  
- Visualization is as important as raw accuracy here.  
- If you’re short on compute, reduce dataset size and epochs — clarity matters more than scale.  

---

Best of luck! See how learning without labels works, and whether it actually helps on real tasks.
