import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def load_and_clean_data(file_path):
    """Load H2 dataset and clean strip strings and invalid null tokens."""
    df = pd.read_csv(file_path)

    # Strip whitespace from string columns and handle pseudo-nulls
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"NULL": np.nan, "Undefined": np.nan, "nan": np.nan})

    return df


def build_preprocessor(numerical_cols, categorical_cols, binary_cols):
    """Build standardized column transformer pipeline."""
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    bin_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipeline, numerical_cols),
        ("cat", cat_pipeline, categorical_cols),
        ("bin", bin_pipeline, binary_cols)
    ])

    return preprocessor


def evaluate_cluster_range(X_transformed, k_range=(2, 7), sample_size=10000, random_state=42):
    """Evaluate KMeans for a range of k values to determine the optimal cluster count."""
    inertias = []
    silhouette_scores = []
    db_scores = []
    ch_scores = []

    # Use a consistent stratified/random sample for silhouette computation on large datasets
    rng = np.random.RandomState(random_state)
    sample_indices = rng.choice(X_transformed.shape[0], size=min(sample_size, X_transformed.shape[0]), replace=False)
    X_sample = X_transformed[sample_indices]

    print(f"Evaluating K-Means for k in range {list(k_range)}...")
    for k in range(k_range[0], k_range[1] + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X_transformed)

        inertia = kmeans.inertia_
        sample_labels = labels[sample_indices]
        sil = silhouette_score(X_sample, sample_labels)
        db = davies_bouldin_score(X_transformed, labels)
        ch = calinski_harabasz_score(X_transformed, labels)

        inertias.append(inertia)
        silhouette_scores.append(sil)
        db_scores.append(db)
        ch_scores.append(ch)

        print(f"  k={k} | Inertia={inertia:.1f} | Silhouette={sil:.4f} | Davies-Bouldin={db:.4f} | Calinski-Harabasz={ch:.1f}")

    metrics_df = pd.DataFrame({
        "k": list(range(k_range[0], k_range[1] + 1)),
        "Inertia": inertias,
        "Silhouette": silhouette_scores,
        "Davies_Bouldin": db_scores,
        "Calinski_Harabasz": ch_scores
    })

    return metrics_df


def plot_evaluation_curves(metrics_df, output_path):
    """Plot Elbow (Inertia) and Silhouette score curves."""
    fig, ax1 = plt.subplots(figsize=(9, 5))

    color = "tab:blue"
    ax1.set_xlabel("Number of Clusters (k)", fontsize=12)
    ax1.set_ylabel("Inertia (Within-Cluster Sum of Squares)", color=color, fontsize=12)
    line1 = ax1.plot(metrics_df["k"], metrics_df["Inertia"], marker="o", color=color, linewidth=2, label="Inertia (Elbow)")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Silhouette Score", color=color, fontsize=12)
    line2 = ax2.plot(metrics_df["k"], metrics_df["Silhouette"], marker="s", color=color, linewidth=2, linestyle="--", label="Silhouette Score")
    ax2.tick_params(axis="y", labelcolor=color)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")

    plt.title("K-Means Model Evaluation: Elbow Method & Silhouette Analysis", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved evaluation plot: {output_path}")


def plot_pca_projections(X_pca, df_visual, output_dir):
    """Plot 2D PCA projections colored by clusters and discrete features."""
    # Plot 1: Colored by Cluster
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        hue=df_visual["Cluster"],
        palette="tab10",
        alpha=0.6,
        s=25
    )
    plt.title("2D PCA Projection: Hotel Booking Clusters", fontsize=14, pad=12)
    plt.xlabel("Principal Component 1 (PC1)", fontsize=12)
    plt.ylabel("Principal Component 2 (PC2)", fontsize=12)
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pca_clusters.png"), dpi=300)
    plt.close()

    # Plot 2: Colored by Discrete Feature (MarketSegment)
    plt.figure(figsize=(10, 7))
    top_segments = df_visual["MarketSegment"].value_counts().nlargest(6).index
    filtered_segment = df_visual["MarketSegment"].where(df_visual["MarketSegment"].isin(top_segments), "Other")

    sns.scatterplot(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        hue=filtered_segment,
        palette="Set2",
        alpha=0.6,
        s=25
    )
    plt.title("2D PCA Projection: Market Segment Distribution", fontsize=14, pad=12)
    plt.xlabel("Principal Component 1 (PC1)", fontsize=12)
    plt.ylabel("Principal Component 2 (PC2)", fontsize=12)
    plt.legend(title="Market Segment", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pca_market_segment.png"), dpi=300)
    plt.close()


def generate_cluster_profiles(df_labeled, numerical_cols, output_dir):
    """Compute and save profile statistics for each cluster."""
    profile_features = [
        "ADR", "LeadTime", "StaysInWeekendNights", "StaysInWeekNights",
        "Adults", "Children", "PreviousCancellations", "BookingChanges",
        "RequiredCarParkingSpaces", "TotalOfSpecialRequests", "IsCanceled"
    ]
    available_features = [f for f in profile_features if f in df_labeled.columns]

    profiles = df_labeled.groupby("Cluster")[available_features].mean().round(2)
    cluster_counts = df_labeled["Cluster"].value_counts().rename("BookingCount")
    cluster_pcts = (df_labeled["Cluster"].value_counts(normalize=True) * 100).round(2).rename("Percentage")

    profiles = pd.concat([cluster_counts, cluster_pcts, profiles], axis=1).sort_index()

    # Save to CSV
    profiles_csv_path = os.path.join(output_dir, "cluster_profiles.csv")
    profiles.to_csv(profiles_csv_path)
    print(f"Saved cluster profiles: {profiles_csv_path}")

    # Plot profile bar charts
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    key_metrics = ["ADR", "LeadTime", "StaysInWeekNights", "TotalOfSpecialRequests", "Adults", "IsCanceled"]

    for i, metric in enumerate(key_metrics):
        if metric in profiles.columns:
            sns.barplot(
                x=profiles.index,
                y=profiles[metric],
                ax=axes[i],
                palette="Blues_d"
            )
            axes[i].set_title(f"Mean {metric} by Cluster", fontsize=12, fontweight="bold")
            axes[i].set_xlabel("Cluster")
            axes[i].set_ylabel(metric)
            axes[i].grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("Cluster Characteristic Comparison Across Key Hotel Metrics", fontsize=15, y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cluster_feature_profiles.png"), dpi=300)
    plt.close()

    return profiles


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, ".."))
    data_path = os.path.join(project_root, "data", "H2.csv")
    output_dir = os.path.join(project_root, "outputs")
    artifact_dir = os.path.join(project_root, "artifacts")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(artifact_dir, exist_ok=True)

    print("=== Step 1: Loading and Cleaning Data ===")
    df = load_and_clean_data(data_path)
    print(f"Dataset shape: {df.shape}")

    # Define feature groups for segmentation
    numerical_columns = [
        "LeadTime", "StaysInWeekendNights", "StaysInWeekNights",
        "Adults", "Children", "Babies", "ADR",
        "PreviousCancellations", "PreviousBookingsNotCanceled",
        "BookingChanges", "DaysInWaitingList",
        "RequiredCarParkingSpaces", "TotalOfSpecialRequests"
    ]

    categorical_columns = [
        "MarketSegment", "DistributionChannel",
        "DepositType", "CustomerType", "Meal"
    ]

    binary_columns = ["IsRepeatedGuest"]

    all_features = numerical_columns + categorical_columns + binary_columns
    X = df[all_features].copy()

    print("=== Step 2: Building Preprocessing Pipeline ===")
    preprocessor = build_preprocessor(numerical_columns, categorical_columns, binary_columns)
    X_transformed = preprocessor.fit_transform(X)
    print(f"Transformed feature matrix shape: {X_transformed.shape}")

    print("=== Step 3: Hyperparameter Search across k Clusters ===")
    metrics_df = evaluate_cluster_range(X_transformed, k_range=(2, 6), sample_size=10000, random_state=42)
    metrics_csv_path = os.path.join(output_dir, "clustering_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    plot_evaluation_curves(metrics_df, os.path.join(output_dir, "elbow_and_silhouette.png"))

    # Choose optimal k based on maximum silhouette score
    best_k = int(metrics_df.loc[metrics_df["Silhouette"].idxmax(), "k"])
    print(f"\nOptimal cluster count selected based on Silhouette score: k = {best_k}")

    print(f"=== Step 4: Training Final K-Means Model (k={best_k}) ===")
    final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    cluster_labels = final_kmeans.fit_predict(X_transformed)

    df["Cluster"] = cluster_labels

    print("=== Step 5: Dimensionality Reduction (PCA) ===")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_transformed)
    var_explained = pca.explained_variance_ratio_
    print(f"PCA Explained Variance: PC1={var_explained[0]:.2%}, PC2={var_explained[1]:.2%}, Total={sum(var_explained):.2%}")

    plot_pca_projections(X_pca, df, output_dir)

    print("=== Step 6: Cluster Profiling and Interpretation ===")
    profiles = generate_cluster_profiles(df, numerical_columns, output_dir)
    print("\nCluster Profiles Summary Table:")
    print(profiles)

    print("=== Step 7: Saving Model Artifacts ===")
    pipeline_bundle = {
        "preprocessor": preprocessor,
        "clustering_model": final_kmeans,
        "pca": pca,
        "features": all_features,
        "best_k": best_k
    }
    artifact_path = os.path.join(artifact_dir, "kmeans_clustering_bundle.joblib")
    joblib.dump(pipeline_bundle, artifact_path)
    print(f"Saved model pipeline bundle to: {artifact_path}")
    print("\n=== Clustering Pipeline Finished Successfully! ===")


if __name__ == "__main__":
    main()

