import gradio as gr
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import math
import io
from PIL import Image
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
DATA_PATH = "data_student.csv"
N_CLUSTERS = 3
DBSCAN_EPS = 1.5
DBSCAN_MIN_SAMPLES = 3
REMOVE_OUTLIERS = True

# ---------- USERS ----------
USERS = {"admin": "1234", "nithya": "password"}

# ---------- Helper Functions ----------
def safe_load_data(path):
    df = pd.read_csv(path)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return df, numeric_cols

def remove_outliers_iqr(df, numeric_cols):
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    df_clean = df[~((df[numeric_cols] < (Q1 - 1.5*IQR)) | (df[numeric_cols] > (Q3 + 1.5*IQR))).any(axis=1)]
    return df_clean

def compute_centroid_dict(X_scaled, labels):
    centroids = {}
    for lbl in np.unique(labels):
        if lbl == -1:
            continue
        mask = labels == lbl
        if mask.sum() > 0:
            centroids[int(lbl)] = X_scaled[mask].mean(axis=0)
    return centroids

def assign_by_nearest_centroid(x_scaled, centroids):
    if len(centroids) == 0:
        return None
    best_lbl, best_dist = None, math.inf
    for lbl, c in centroids.items():
        dist = np.linalg.norm(x_scaled.ravel() - c)
        if dist < best_dist:
            best_dist = dist
            best_lbl = int(lbl)
    return best_lbl

def plot_selected_algorithm(X_pca, labels, x_input_pca, pred_label, algo_name):
    fig, ax = plt.subplots(figsize=(6,5))
    unique_labels = np.unique(labels)
    cmap = plt.cm.get_cmap("tab10")
    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        color = cmap(i % 10) if lbl != -1 else (0.6,0.6,0.6)
        ax.scatter(X_pca[mask,0], X_pca[mask,1], c=[color], s=40, label=f"Cluster {lbl}" if lbl!=-1 else "Outlier", alpha=0.6, edgecolors='k', linewidths=0.2)
    ax.scatter(x_input_pca[0,0], x_input_pca[0,1], c='red', marker='X', s=200, label='Your Input', edgecolors='k')
    ax.set_title(f"{algo_name} (pred: {pred_label})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc='best', fontsize=8)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

# ---------- Load & preprocess ----------
df, numeric_cols = safe_load_data(DATA_PATH)
if REMOVE_OUTLIERS:
    df = remove_outliers_iqr(df, numeric_cols)

X = df[numeric_cols].values.astype(float)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

hier_labels = AgglomerativeClustering(n_clusters=N_CLUSTERS).fit_predict(X_scaled)
hier_centroids = compute_centroid_dict(X_scaled, hier_labels)

db_labels = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(X_scaled)
db_centroids = compute_centroid_dict(X_scaled, db_labels)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# ---------- Gradio Functions ----------
def check_login(username, password):
    return username in USERS and USERS[username] == password

def predict_selected_algo(selected_algo, highlight_outliers, *user_inputs):
    user_arr = np.array([user_inputs], dtype=float)
    user_scaled = scaler.transform(user_arr)
    user_pca = pca.transform(user_scaled)

    pred_label = None
    plot_img = None

    if selected_algo == 'KMeans':
        pred_label = int(kmeans.predict(user_scaled)[0])
        label_map = {0: " cluster 0:Low", 1: "cluster 1:Middle", 2: "cluster2:High"}
        cluster_name = label_map.get(pred_label, str(pred_label))

        plot_img = plot_selected_algorithm(
            X_pca, kmeans_labels, user_pca, cluster_name, "KMeans"
        )
        return cluster_name, plot_img
    elif selected_algo == 'Hierarchical':
        pred_label = assign_by_nearest_centroid(user_scaled, hier_centroids)
        plot_img = plot_selected_algorithm(X_pca, hier_labels, user_pca, pred_label, "Hierarchical")
    elif selected_algo == 'DBSCAN':
        pred_label = assign_by_nearest_centroid(user_scaled, db_centroids)
        if pred_label is None:
            pred_label = -1
        if highlight_outliers:
            db_labels_mapped = np.array(["Outlier" if l == -1 else f"Cluster {l}" for l in db_labels])
            plot_img = plot_selected_algorithm(X_pca, db_labels_mapped, user_pca,
                                               "Outlier" if pred_label==-1 else f"Cluster {pred_label}", "DBSCAN")
        else:
            plot_img = plot_selected_algorithm(X_pca, db_labels, user_pca, pred_label, "DBSCAN")

    return str(pred_label), plot_img

# ---------- Build Gradio App ----------
with gr.Blocks() as demo:
    # Login UI
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 🔐 Login")
            username_input = gr.Textbox(label="Username")
            password_input = gr.Textbox(label="Password", type="password")
            login_btn = gr.Button("Login")
            login_message = gr.Textbox(label="Status")

    # Main UI
    main_ui = gr.Column(visible=False)
    with main_ui:
        gr.Markdown("### Choose Algorithm:")
        algo_select = gr.Dropdown(["KMeans","Hierarchical","DBSCAN"], value="KMeans", label="Select Algorithm")
        highlight_outliers = gr.Checkbox(label="Highlight Outliers (DBSCAN only)", value=True)
        gr.Markdown("### Enter Numeric Features:")
        input_fields = [gr.Number(label=col) for col in numeric_cols]
        predict_btn = gr.Button("Predict Cluster")
        cluster_out = gr.Textbox(label="Predicted Cluster")
        plot_out = gr.Image(label="Cluster Plot")

    # Login Action
    def login_action(username, password):
        if check_login(username, password):
            return "Login successful! You can now use the clustering tool.", gr.update(visible=True)
        else:
            return "Login failed! Check username/password.", gr.update(visible=False)

    login_btn.click(fn=login_action, inputs=[username_input, password_input], outputs=[login_message, main_ui])

    # Predict Action
    predict_btn.click(fn=predict_selected_algo, inputs=[algo_select, highlight_outliers]+input_fields,
                      outputs=[cluster_out, plot_out])

# ---------- Launch ----------
if __name__ == "__main__":
    demo.launch(share=True)
