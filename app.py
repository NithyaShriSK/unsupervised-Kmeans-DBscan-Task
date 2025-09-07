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
USERS = {
    "admin": "1234",
    "nithya": "password"
}

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
    unique = np.unique(labels)
    for lbl in unique:
        if lbl == -1:
            continue
        mask = labels == lbl
        if mask.sum() > 0:
            centroids[int(lbl)] = X_scaled[mask].mean(axis=0)
    return centroids

def assign_by_nearest_centroid(x_scaled, centroids):
    if len(centroids) == 0:
        return None
    best_lbl = None
    best_dist = math.inf
    for lbl, c in centroids.items():
        d = np.linalg.norm(x_scaled.ravel() - c)
        if d < best_dist:
            best_dist = d
            best_lbl = int(lbl)
    return best_lbl

def create_combined_plot(X_pca, labels_dict, x_input_pca, pred_labels):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    algos = ['KMeans', 'Hierarchical', 'DBSCAN']
    cmap = plt.cm.get_cmap("tab10")
    for i, algo in enumerate(algos):
        ax = axes[i]
        lbls = labels_dict[algo]
        unique = np.unique(lbls)
        for j, u in enumerate(unique):
            mask = lbls == u
            color = cmap(j % 10) if u != -1 else (0.6,0.6,0.6)
            ax.scatter(X_pca[mask,0], X_pca[mask,1], c=[color], s=40, label=f"Cluster {u}", alpha=0.6, edgecolors='k', linewidths=0.2)
        ax.scatter(x_input_pca[0,0], x_input_pca[0,1], c='red', marker='X', s=200, label='Your Input', edgecolors='k')
        ax.set_title(f"{algo} (pred: {pred_labels.get(algo,'N/A')})")
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

labels_dict_train = {
    'KMeans': kmeans_labels,
    'Hierarchical': hier_labels,
    'DBSCAN': db_labels
}

# ---------- Gradio Functions ----------
def check_login(username, password):
    if username in USERS and USERS[username] == password:
        return True
    return False

def predict_and_visualize(*user_inputs):
    user_arr = np.array([user_inputs], dtype=float)
    user_scaled = scaler.transform(user_arr)

    km_label = int(kmeans.predict(user_scaled)[0])
    hier_label = assign_by_nearest_centroid(user_scaled, hier_centroids)
    db_label = assign_by_nearest_centroid(user_scaled, db_centroids)
    if db_label is None:
        db_label = -1

    user_pca = pca.transform(user_scaled)
    pred_labels = {'KMeans': km_label, 'Hierarchical': hier_label, 'DBSCAN': db_label}

    img = create_combined_plot(X_pca, labels_dict_train, user_pca, pred_labels)
    return f"{km_label}", f"{hier_label}", f"{db_label}", img

# ---------- Build Gradio App ----------
with gr.Blocks() as demo:
    # ---------- Login UI ----------
    with gr.Row():
        with gr.Column():
            gr.Markdown("## 🔐 Login")
            username_input = gr.Textbox(label="Username")
            password_input = gr.Textbox(label="Password", type="password")
            login_btn = gr.Button("Login")
            login_message = gr.Textbox(label="Status")

    # ---------- Main Clustering UI (hidden initially) ----------
    with gr.Row(visible=False) as main_ui:
        gr.Markdown("# 🧮 Customer Clustering Explorer")
        gr.Markdown("Enter numeric features below and predict clusters:")
        input_fields = [gr.Number(label=col) for col in numeric_cols]
        predict_btn = gr.Button("Predict Clusters")
        km_out = gr.Textbox(label="KMeans Cluster")
        hier_out = gr.Textbox(label="Hierarchical Cluster")
        db_out = gr.Textbox(label="DBSCAN Cluster")
        plot_out = gr.Image(label="Cluster visualization (PCA 2D)")

        predict_btn.click(fn=predict_and_visualize,
                          inputs=input_fields,
                          outputs=[km_out, hier_out, db_out, plot_out])

    # ---------- Login Action ----------
    def login_action(username, password):
        if check_login(username, password):
            return "Login successful! You can now use the clustering tool.", gr.update(visible=True)
        else:
            return "Login failed! Please check your username and password.", gr.update(visible=False)

    login_btn.click(fn=login_action,
                    inputs=[username_input, password_input],
                    outputs=[login_message, main_ui])


# ---------- Launch ----------
if __name__ == "__main__":
    demo.launch(share=True)
