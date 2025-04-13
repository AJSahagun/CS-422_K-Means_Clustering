<!-- Intro-->

<!--
* Thanks for reviewing my Project-README-Template! 
* Access the blank-template here (https://github.com/YousefIbrahimismail/Project-README-Template/blob/main/Templates/_blank-README.md) 
* 
* Read the comments for an easy step by step guide.or read this Make_it_Yours guide here: () // add Personalization_md_file
* Enjoy!
-->


<!-- Shields Section--><!-- Optional -->

<!-- 
* Insert project shields and badges through this link https://shields.io/
* 
*
-->

<div align="center">
    <a href="https://github.com/AJSahagun/CS-422_K-Means_Clustering/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/AJSahagun/CS-422_K-Means_Clustering?color=yellow&label=Project%20Stars&style=for-the-badge"></a>
    <a href="https://github.com/AJSahagun/CS-422_K-Means_Clustering/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/AJSahagun/CS-422_K-Means_Clustering?color=brightgreen&label=issues&style=for-the-badge"></a>
    <a href="https://github.com/AJSahagun/CS-422_K-Means_Clustering/network"><img alt="GitHub forks" src="https://img.shields.io/github/forks/AJSahagun/CS-422_K-Means_Clustering?color=9cf&label=forks&style=for-the-badge"></a>
</div>
<br>


<!-- Logo Section  --><!-- Required -->

<!--
* Insert an image URL in the <img> "src" attribute bellow. (line )
* 
* Insert your github profile URL in the <a> "href" attribute bellow (line )
-->


<div align="center">
    <a href="https://github.com/AJSahagun/" target="_blank">
        <img src="./img/logo.jpg" 
        alt="Logo" height="290">
    </a>
</div>


<!-- Project title 
* use a dynamic typing-SvG here https://readme-typing-svg.demolab.com/demo/
*
*  Instead you can type your project name after a # header
-->

<div align="center">
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=4000&pause=5000&background=FFFFFF00&center=true&vCenter=true&multiline=true&width=435&lines=K-Means Clustering [Group 4]">
</div>


## About<!-- Required -->
<!-- 
* information about the project 
* 
* keep it short and sweet
-->


This project demonstrates K-means clustering on the Iris dataset using petal length and width features. The implementation includes data preprocessing, model training, evaluation, and visualization of results. Key components:

- **Data Preprocessing**: Scales features and maps cluster labels to true species.
- **Model Training**: Uses the elbow method to determine optimal clusters (fixed at k=3 for Iris).
- **Visualization**: Generates plots for the elbow method, clustering results, true species distribution, and a confusion matrix.
- **Animation**: Version 3 includes an animated visualization showing the evolution of K-means clustering in 3D space.


## How to use this project<!-- Required -->
<!-- 
* Here you may add information about how 
* 
* and why to use this project.
-->

- Clone this repository into your local machine.

```bash
    git clone https://github.com/AJSahagun/CS-422_K-Means_Clustering.git
```

- Set up your environment
```bash
    pip install -r requirements.txt
```

- Change the import path on `main.py` based on what version you will run
```python
from src.ver_1.preprocessing import load_and_preprocess_data
. . .
from src.ver_1.visualization import plot_clustering_analysis, calculate_wcss, find_elbow_point
```

Run the main script:
- For Version 1 and 2
```bash
    python main1-2.py
```
- For Version 3
```bash
    python main3.py
```

## Demo<!-- Required -->
<!-- 
* You can add a demo here GH supports images/ GIFs/videos 
* 
* It's recommended to use GIFs as they are more dynamic
-->

### Version 1 - Features: Petal Length & Width (2D)
<img src="./img/visualization1.png" alt="Visualization for version 1" height="290">

```commandline
Model Evaluation Metrics
Optimal K: 2
Precision: 0.503
Recall: 0.670
F1-Score: 0.559
```

### Version 2 - Features: Sepal Length & Width (2D)
<img src="./img/visualization2.png" alt="Visualization for version 2" height="290">

```commandline
Model Evaluation Metrics
Optimal K: 2
Precision: 0.503
Recall: 0.670
F1-Score: 0.559
```

### Version 3 - Features: Petal Length & Width, Sepal Width (3D)
<img src="./img/visualization3.png" alt="Visualization for version 3" height="290">

```commandline
K-means converged after 5 iterations

Model Evaluation Metrics
Optimal K: 2
Precision: 0.503
Recall: 0.670
F1-Score: 0.559
```

### Version 3 - K-means Iteration Visualization
<img src="./img/kmeans_iterations.png" alt="K-means iterations visualization" height="290">

### Version 3 - K-means Animation
<img src="./img/kmeans_animation.gif" alt="K-means animation" height="290">

### Output

- A 2x2 visualization grid:
  1. Elbow Method Plot 
  2. K-means Clustering Results 
  3. True Species Distribution 
  4. Confusion Matrix
- Iteration visualizations showing clustering progress (New in Version 3)
- Animated visualization of K-means algorithm (New in Version 3)
- Printed evaluation metrics (Optimal k, precision, recall, F1-score).

<div align="center">

[//]: # (    <img alt="demo" src="https://user-images.githubusercontent.com/59213365/198179320-b9ec79a6-b5ca-47b9-9f29-125d18f6be70.gif">)
</div>

## Table of Contents<!-- Optional -->
<!-- 
* This section is optional, yet having a contents table 
* helps keeping your README readable and more professional.
* 
* If you are not familiar with HTML, no worries we all been there :) 
* Review learning resources to create anchor links. 
-->


<dev display="inline-table" vertical-align="middle">
<table align="center" vertical-align="middle">
        <tr>
            <td><a href="#about">About</a></td>        
            <td><a href="#how-to-use-this-project">Getting started</a></td>
            <td><a href="#demo">Demo</a></td>
            <td><a href="#project-roadmap--">Project Roadmap</a></td>
        </tr>
        <tr>
            <td><a href="#documentation">Documentation</a></td>
            <td><a href="#contributors">Contributors</a></td>
            <td><a href="#acknowledgments">Acknowledgments</a></td>
            <td><a href="#contact">Contact</a></td>
        </tr>
</table>
</dev>


<!-- - Use this html element to create a back to top button. -->
<p align="right"><a href="#how-to-use-this-project">back to top ⬆️</a></p>


## Project Roadmap <!-- Optional --> <!-- add learning_Rs-->
<!-- 
* Add this section in case the project has different phases
* 
* Under production or will be updated.
-->

- [x] Create Project Repository
- [x] Choose a dataset
- [x] Start ML Model development
- [x] Evaluate the Model's accuracy
- [x] Implement visualization
- [x] Implement Revisions
- [x] Add animated visualization of K-means process


<p align="right"><a href="#how-to-use-this-project">back to top ⬆️</a></p>

## Documentation<!-- Optional -->
<!-- 
* You may add any documentation or Wikis here
* 
* 
-->
### Project Structure
```
CS-422_K-Means_Clustering/
│
├── data/
│   ├── Iris-Original.csv
│   └── ml_group-4_iris_dataset.csv
│
├── src/
│   ├── ver_1/                 # Features: Petal Length & Width (2D)
│   │   ├── preprocessing.py
│   │   └── visualization.py
│   ├── ver_2/                 # Features: Sepal Length & Width (2D)
│   │   ├── preprocessing.py
│   │   └── visualization.py
│   ├── ver_3/                 # Features: Petal Length & Width, Sepal Width (3D)
│   │   ├── preprocessing.py
│   │   └── visualization.py
│   ├── custom_kmeans.py       # KMeans model implementation with iteration tracking
│   └── kmeans_model.py        # KMeans model from Sci-kit
│   
│
├── notebook/
│   └── kmeans.ipynb           # Jupyter notebook for the model
│
├── img/
│   ├── logo.jpg               # Picture logo used in Read me
│   ├── visualization1.png     # Demo picture for version 1
│   ├── visualization2.png     # Demo picture for version 2
│   ├── visualization3.png     # Demo picture for version 3
│   ├── kmeans_iterations.png  # K-means iteration steps visualization
│   └── kmeans_animation.gif   # Animated visualization of K-means process
│
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
├── main1-2.py                 # Main script to run version 1 & 2
└── main3.py                   # Main script to run version 3
```

### Module Documentation

#### `main3.py`
**Entry point for Version 3:**
1. Loads and preprocesses data 
2. Calculates WCSS for elbow method 
3. Trains model with optimal k=3 
4. Evaluates performance 
5. Generates composite visualization
6. Shows iteration-by-iteration progress of K-means
7. Creates and displays animated visualization of clustering process

#### `preprocessing.py`
**Data handling:**
- load_and_preprocess_data(file_path)
- Returns: (X_scaled, y, scaler)
  - Loads data 
  - Extracts petal features 
  - Scales features using StandardScaler
- create_label_mapping(true_labels, cluster_labels)
- Returns: Best label mapping dictionary 
  - Finds optimal cluster-to-species mapping using permutation accuracy tests

#### `custom_kmeans.py`
**Custom clustering implementation:**
- KMeans class: (Core algorithm with)
  - Random centroid initialization 
  - Iterative centroid updating 
  - Convergence checking 
  - WCSS calculation
  - New: Tracking centroids and labels for each iteration
- IrisKMeans wrapper class: (Methods)
  - fit(): Train model 
  - predict(): Get cluster labels 
  - evaluate(): Returns metrics dictionary with:
    - Confusion matrix 
    - Precision/Recall/F1-score
  - get_iteration_history(): Returns tracked centroids and labels history

#### `visualization.py`
**Plotting functions:**
- plot_clustering_analysis(): (Generates 4-panel figure showing)
  1. Elbow method curve 
  2. Cluster assignments with centroids
  3. True species distribution
  4. Confusion matrix
- calculate_wcss():
  - Computes WCSS values for k=1 to max_k
- New: plot_kmeans_iterations():
  - Creates a series of 3D plots showing how clusters evolve in each iteration
- New: create_kmeans_animation():
  - Creates an animated visualization showing the progressive movement of centroids
  - Shows real-time assignment of points to clusters


## Contributors<!-- Required -->
<!-- 
* Without contribution we wouldn't have open source. 
* 
* Generate github contributors Image here https://contrib.rocks/preview?repo=angular%2Fangular-ja
-->

<a href="https://github.com/AJSahagun/CS-422_K-Means_Clustering/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=AJSahagun/CS-422_K-Means_Clustering" />
</a>


## Acknowledgments<!-- Optional -->
<!-- 
* Credit where it's do 
* 
* Feel free to share your inspiration sources, Stackoverflow questions, github repos, tools etc.
-->
### Group Members
#### Data Gathering
- CARL JUSTIN ULARTE
- RONALDO PESIGAN
#### Model Development
- MICHAEL JOHN RHEY MASAGNAY
- AARON JOHN SAHAGUN
- ROSELER TANEDO JR
#### Documentation
- JOHN PAUL MEDRANO
- JOHN HAROLD PRIETO
- DESXZOR NAVARRO
- CAREN TOLENTINO


<!-- - Use this html element to create a back to top button. -->
<p align="right"><a href="#how-to-use-this-project">back to top ⬆️</a></p>


## Feedback<!-- Required -->
<!-- 
* You can add contacts information like your email and social media account 
* 
* Also it's common to add some PR guidance.
-->


> You can make this project better, please  feel free to open a [Pull Request](https://github.com/AJSahagun/CS-422_K-Means_Clustering/pulls).
- If you notice a bug or a typo use the tag **"Correction"**.
- If you want to share any ideas to help make this project better, use the tag **"Enhancement"**.

<details>
    <summary>Contact Me 📨</summary>

### Contact<!-- Required -->
Reach me via email: [21-04332@g.batstate-u.edu.ph](mailto:21-04332@g.batstate-u.edu.ph)
<!-- 
* add your email and contact info here
* 
* 
-->
    
</details>


<!-- - Use this html element to create a back to top button. -->
<p align="right"><a href="#how-to-use-this-project">back to top ⬆️</a></p>