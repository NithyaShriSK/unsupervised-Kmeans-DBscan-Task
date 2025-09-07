# Unsupervised Clustering with KMeans, Hierarchical, and DBSCAN

This project demonstrates **unsupervised clustering** using three algorithms: **KMeans**, **Hierarchical Clustering**, and **DBSCAN**. It allows users to analyze datasets and visualize clusters in 2D using **PCA**.

---

## Features

- Load a dataset (CSV) containing numeric features
- Remove outliers using IQR method
- Apply **KMeans**, **Hierarchical**, and **DBSCAN** clustering
- Scale data using **StandardScaler**
- Visualize clusters in 2D using **PCA**
- Predict cluster for new input points
- User-friendly **Gradio web interface** with login functionality

---

## GitHub Repository

You can access the complete notebook and code here:

[Unsupervised KMeans & DBSCAN Task Notebook](https://github.com/NithyaShriSK/unsupervised-Kmeans-DBscan-Task/blob/main/cluster.ipynb)

---

## Hugging Face Space

Try the interactive clustering app online here:

[KMeans & DBSCAN Task on Hugging Face](https://huggingface.co/spaces/NithyaShriSK/KmeansDbscanTask)

---

## How to Use

1. **Login** using one of the predefined usernames and passwords:  
   - Username: `admin`, Password: `1234`  
   - Username: `nithya`, Password: `password`  

2. **Upload your CSV file** (must contain numeric features)  
3. **Train the clustering model** using KMeans, Hierarchical, and DBSCAN  
4. **Enter new data points** to predict clusters  
5. **View 2D PCA visualization** of clusters and input points

---

## Visualization Images

You can include example outputs from the `images/` folder like this:

![Cluster Example 1](images/input.png)
![Cluster Example 2](images/graph.png)


---

## Requirements

- Python 3.8+  
- pandas  
- numpy  
- scikit-learn  
- matplotlib  
- Pillow  
- gradio  

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## License

This project is open-source and available under the MIT License.

