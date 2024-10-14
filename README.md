# Customer Segmentation with Clustering Techniques

## Overview

This project aims to perform customer segmentation using clustering techniques on a dataset related to marketing campaigns. The goal is to analyze customer behavior, group customers into distinct segments, and provide actionable insights for targeted marketing strategies. The dataset is preprocessed, followed by feature engineering, exploratory data analysis (EDA), dimensionality reduction, clustering, and visualization to derive meaningful conclusions.

## Dataset

The dataset used in this project contains information about customers' demographics, purchasing behavior, and marketing responses. The key features include:

- **Age**: Age of the customer.
- **Income**: Annual income of the customer.
- **Education**: Educational level.
- **Family Size**: Number of family members.
- **Spending**: Total spending on various products.
- **Children**: Number of children in the household.
- **Customer For**: Number of days the customer has been associated with the company.

The dataset also includes features related to product spending, such as:
- **Wines, Fruits, Meat, Fish, Sweets, Gold Products**: Amounts spent on these product categories.

## Project Steps

### 1. **Importing Libraries**
The project utilizes various Python libraries for data analysis, visualization, and machine learning. Libraries include:
- **Pandas**: Data manipulation and analysis.
- **Numpy**: Mathematical operations.
- **Matplotlib & Seaborn**: Visualization tools.
- **Scikit-learn**: Machine learning and clustering.
- **Plotly**: Interactive visualizations.
- **Yellowbrick**: Visualizing clustering results.

### 2. **Data Loading and Exploration**
The data is loaded from a `.csv` file, and an initial exploration is performed to understand its structure, including checking data types, missing values, and basic statistics.

```python
data = load_data('marketing_campaign.csv')
explore_dataset(data)
```

### 3. **Data Cleaning**
- **Handling Missing Values**: Missing income values are imputed with the median, and rows with missing values in other columns are dropped.
- **Outlier Removal**: Rows with unrealistic ages or incomes are removed to ensure data quality.
- **Date Parsing**: The customer's enrollment date is parsed, and the duration of association with the company is calculated.

```python
data = handle_missing_values(data)
data = parse_dates(data)
data = remove_outliers(data)
```

### 4. **Feature Engineering**
New features are created to enhance the dataset, including:
- **Customer For**: Number of days since customer enrollment.
- **Is Parent**: Indicator of whether the customer has children.
- **Income per Child**: Adjusted income based on the number of children.
- **Spent per Family Size**: Total spending normalized by family size.

```python
data = feature_engineering(data)
```

### 5. **Exploratory Data Analysis (EDA)**
Pairwise relationships between selected features are visualized, and the correlation matrix is plotted to understand feature interactions.

```python
initial_visualization(data)
plot_correlation_matrix(data)
```

### 6. **Data Preprocessing**
- **Categorical Encoding**: Categorical variables are one-hot encoded.
- **Feature Scaling**: All features are scaled to ensure that they are on the same scale, which is necessary for clustering.

```python
data = encode_categorical(data)
scaled_data = scale_features(data)
```

### 7. **Dimensionality Reduction**
- **PCA**: Principal Component Analysis is applied to reduce the dimensionality of the dataset while retaining 3 key components.
- **Visualization**: Both static and interactive 3D visualizations of PCA results are created.

```python
pca_df = apply_pca(scaled_data, n_components=3)
plot_pca_3d(pca_df)
interactive_plot(pca_df)
```

### 8. **Clustering**
- **Elbow Method**: The optimal number of clusters is determined using the elbow method.
- **Agglomerative Clustering**: Hierarchical clustering is applied, and clusters are assigned to the data.
- **KMeans**: Alternatively, KMeans clustering can also be applied within a pipeline for efficient scaling and clustering.

```python
optimal_clusters = elbow_method(pca_df)
clusters = agglomerative_clustering(pca_df, optimal_clusters)
```

### 9. **Evaluation and Visualization**
- **Silhouette Score**: The silhouette score is used to evaluate the quality of the clustering.
- **Cluster Visualization**: 3D scatter plots of clusters, both static and interactive, are generated to visually assess cluster separation.

```python
evaluating_clustering(pca_df, clusters)
plot_clusters_3d(pca_df)
```

### 10. **Cluster Profiling**
Each cluster is analyzed in detail to understand its characteristics. Key insights, such as age, income, spending patterns, and family size, are derived to inform marketing strategies.

```python
cluster_profiling(data)
```

### 11. **Advanced Analysis**
Multiple clustering algorithms are compared, and cluster stability is assessed through repeated runs. This provides insights into the robustness of the clustering results.

```python
compare_clustering_algorithms(pca_df.drop('Cluster', axis=1), optimal_clusters)
cluster_stability(scaled_data, optimal_clusters, n_runs=10)
```

## Results
The project concludes with detailed cluster profiles and actionable insights. For example, clusters may represent high-income families, young customers with low spending, or large families with high product usage. These insights can guide targeted marketing campaigns.

## How to Run the Project

1. Install the necessary Python libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn plotly yellowbrick
   ```

2. Run the Python script or Jupyter Notebook. Ensure the dataset (`marketing_campaign.csv`) is in the correct directory.

3. The results, including visualizations and clustering insights, will be displayed.

## Conclusion
This project demonstrates a comprehensive approach to customer segmentation using clustering techniques. By identifying customer groups with similar behaviors, businesses can create more personalized marketing strategies and improve customer satisfaction.
