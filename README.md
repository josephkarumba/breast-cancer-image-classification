# IDENTIFICATION OF BREAST CANCER FROM HISTOPATHOLOGICAL IMAGES IN KENYA


![image](https://github.com/user-attachments/assets/0513a380-f5bf-4be5-94a4-12d0cf9f2284)


# **INTRODUCTION** 
Breast cancer stands as a formidable public health challenge in Kenya, characterized by alarmingly high mortality rates largely attributed to late-stage detection. The country's healthcare system grapples with limited resources, including a shortage of skilled pathologists and a dearth of advanced diagnostic technologies. Consequently, the prevailing manual diagnostic methods are time-consuming, error-prone, and often lead to delayed or inaccurate diagnoses. This project aims to address these critical issues by developing and implementing a robust machine learning model capable of accurately classifying breast cancer types from histopathological images.


# **BUSINESS UNDERSTANDING**
The overarching goal of this project is to address the critical issue of breast cancer in Kenya, characterized by high mortality rates due to late-stage detection and limited access to quality healthcare. By developing a deep machine learning-based solution, this aims to enhance early diagnosis, improve diagnostic efficiency, and expand access to accurate breast cancer screening services. Ultimately, this project seeks to contribute to a significant reduction in breast cancer mortality rates and improve the overall quality of life for Kenyan women.

# **PROBLEM STATEMENT**
Breast cancer poses a significant public health challenge, characterized by high mortality rates primarily due to late-stage detection. The country's healthcare system faces numerous obstacles in addressing this crisis, including limited access to specialized healthcare, a shortage of skilled pathologists, and a lack of advanced diagnostic technologies. Consequently, the current manual diagnostic process is time-consuming, error-prone, and often leads to delayed or inaccurate diagnoses. This delay in identifying and treating breast cancer has severe implications for patient outcomes, including increased morbidity and mortality rates. To mitigate these challenges and improve breast cancer care in Kenya, there is a critical need for innovative solutions that can enhance early detection, improve diagnostic accuracy, and increase accessibility to quality care.

# **OBJECTIVES**
1. Develop a robust image classification model: Create an accurate and efficient machine learning model capable of distinguishing between benign and malignant breast tissue based on histopathological images.
2. Improve diagnostic efficiency: Accelerate the diagnostic process by automating image analysis tasks, reducing the workload on pathologists, and enabling faster patient treatment.
3. Enhance patient outcomes: Contribute to early detection of breast cancer, leading to improved treatment options and increased survival rates.

**Goals**

* Analyze existing data on breast cancer in Kenya.
* Develop a machine learning model for classifying histopathological images.
* Evaluate the model's performance and potential impact.
* Explore integration strategies for the model into Kenya's healthcare system.

# **METRICS OF SUCCESS**
1. Accuracy: As a measure of the proportion of correctly predicted instances out of the total instances. It reflects the model's ability to classify data points accurately, with higher accuracy indicating better performance in distinguishing between different classes.
2. Loss: As the measure of how well the model's predictions match the actual values. It quantifies the difference between predicted and true values, with a lower loss indicating a more accurate model. Evaluating the test loss helps determine how well the model can generalize to new, unseen data.

**Expected Outcomes**

* A better understanding of the breast cancer landscape in Kenya.
* A machine learning model capable of accurately classifying breast cancer types.
* Recommendations for implementing the model in Kenyan healthcare settings.

## Methodology

### Data Collection
The BreakHis dataset, a publicly available dataset of histopathological images of breast tissue, was used for this project. 
The dataset consists of 7,909 images, divided into two classes: benign (2,480 images) and malignant (5,429 images). this is inclusive of the X40, X100, X200, X400 Magnification levels for the main categories and their subcategories.

(https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/)

## Data Preparation & Cleaning
* Eliminate duplicates and irrelevant images to reduce noise and improve model performance.
* Enhance images by filtering, sharpening, or adjusting contrast to improve clarity.
* Crop images to focus on relevant subjects and resize them to a consistent dimension.
* Conduct manual checks to verify the accuracy of images and labels, ensuring quality control.
* Create a structured directory for images, organizing them into labeled folders for better management.
* Review and correct labels associated with images to ensure accuracy and consistency.

## Data Description & Structure
### Benign Category
1. Adenosis: 113 - a non-cancerous condition where the breast lobules are enlarged.
2. Fibroadenoma: 260 - a common benign breast tumor made up of glandular and stromal tissue.
3. Phyllodes Tumor: 121 -  are rare breast tumors that can be benign but have the potential to become malignant.
4. Tubular Adenoma: 150 -  a benign breast tumor that resembles the milk ducts.

### Malignant Category
1. Ductal Carcinoma: 903 -The largest subset, ductal carcinoma refers to cancer that begins in the milk ducts and is one of the most common types of breast cancer.
2. Lobular Carcinoma: 170 -  a type of cancer that begins in the lobules (milk-producing glands) of the breast.
3. Mucinous Carcinoma: 172 -  a rare type of breast cancer characterized by the production of mucin.
4. Papillary Carcinoma: 142 -  a rare form of breast cancer that has finger-like projections.

![Sample-of-distinct-types-of-breast-cancer-histopathology-images-from-the-BreaKHis-dataset](https://github.com/user-attachments/assets/0698d93f-ba4c-4e26-b874-a8a0c2a9c925)


# **Explorative Data Analysis (EDA)**

![image](https://github.com/user-attachments/assets/d929b278-7877-412c-b774-1e33653a5f0f)

The dataset exhibits class imbalance, with a significantly higher number of malignant cases compared to benign cases.  The chart clearly shows the predominance of ductal carcinoma cases compared to other categories. This suggests that ductal carcinoma is the most prevalent type of breast cancer in the dataset. 



![image](https://github.com/user-attachments/assets/6436d6e2-a693-4f3d-918f-cc61cb18df67)


The chart illustrates the distribution of image heights within the dataset. The majority of images fall within the 450-500 pixel height range, with a decreasing frequency for both smaller and larger image heights.

![image](https://github.com/user-attachments/assets/01ead559-32a0-4f0a-9420-e6308736eab0)

The plot exhibits a skewed distribution with a rightward tail.  The distribution has a single peak around the pixel intensity value of 0.65. This suggests that the majority of pixels in the lobular carcinoma images have a similar intensity level. The pixel intensity values range from approximately 0.4 to 0.9,  with a limited range of brightness. The skewed distribution suggests that lobular carcinoma images may have a characteristic pattern of intensity values.
The single peak indicates a relatively homogeneous intensity distribution within the images.

![image](https://github.com/user-attachments/assets/6da1bc3a-3cbf-46cd-9117-8b8dc46eff2d)

The plot exhibits a skewed distribution, with a longer tail on the right side. This suggests that a larger proportion of pixels in papillary carcinoma images have higher intensity values compared to lower intensity values. The distribution has a single peak around the pixel intensity value of 0.7. This indicates that most pixels in these images fall within this intensity range.
Range: The pixel intensity values range from approximately 0.5 to 0.9, The skewed distribution and single peak suggest that papillary carcinoma images may have a characteristic pattern of intensity values, potentially related to the unique features of this cancer type. 

![image](https://github.com/user-attachments/assets/0a0e9162-30a6-4120-9dfc-962974bdd2fe)

The analysis of image aspect ratios reveals a consistent rectangular format for most images in the dataset. The majority of images have an aspect ratio close to 1.522, indicating a slightly wider width than height. While there are a few exceptions with an aspect ratio of 1.535, these variations are less common. Overall, the dataset exhibits a predominantly standardized image format.


![image](https://github.com/user-attachments/assets/b7ca4458-cfdb-4ed9-ad00-93550bce0721)



t-SNE (t-Distributed Stochastic Neighbor Embedding) is a dimensionality reduction technique used to visualize high-dimensional data in a lower-dimensional space while preserving the underlying structure of the data. 
Each point in the plot represents an image, and its position is determined by its calculated t-SNE components. Green points represent benign images, and red points represent malignant images. the t-SNE algorithm has effectively preserved the underlying structure of the data. indicating that the image features associated with these two classes are different.

## Data Preprocessing
1. Resizing images to a consistent dimension is essential for feeding them into neural networks, which require fixed input sizes.
2. Normalization & Scaling by adjusting the pixel values of images to a common scale.
3. Data Augmentation artificially increase the size of the training dataset by creating variations of existing images.
4. Batching the dataset into smaller subsets that are processed together during training.
5. Combine all benign images into a single folder and all malignant images into a different folder for clear organization.


### Statistical Analysis
Hypothesis Testing
Hypothesis testing tested to validate assumptions about the dataset and draw inferences. It assesed the performance of different models or algorithms (comparing accuracy between a baseline model and a new model).It validated assumptions about the distribution of pixel values or class labels, ensuring that the data met the necessary conditions for modeling techniques.

 Correlation Analysis
Correlation analysis assessed the strength and direction of relationships between variables, feature selection and modeling strategies.
Identified relationships between different features (e.g. pixel intensity values, color channels) and the target variable ( class labels), guiding the selection of relevant features for the model.


# MODELING

### Model Architecture

1. DenseNet (DenseNet)  DenseNet introduces dense connections between layers, where the output of each layer is concatenated with the outputs of all preceding layers. This encourages feature reuse and reduces the vanishing gradient problem.
2. Convolutional Neural Network (CNN)
 CNNs are specifically designed for processing and analyzing image data. They consist of convolutional layers, pooling layers, and fully connected layers.
4. VGG16 - VGG16 is a deep CNN architecture with multiple convolutional layers and pooling layers. It is characterized by its use of small (3x3) convolutional filters stacked on top of each other.

### Model Training

Models were trained using a combination of training and validation datasets. We used optimization algorithms to minimize the loss function and update model parameters. Hyperparameters such as learning rate, batch size, and number of epochs were tuned for optimal performance.

**Model Evaluation**
------------------

* **Accuracy**: Overall proportion of correct predictions.
* **Precision**: Proportion of positive predictions that were correct.
* **Recall (Sensitivity)**: Proportion of actual positive cases correctly identified.
* **Specificity**: Proportion of actual negative cases correctly identified (primarily for ResNet model).
* **F1-score**: Harmonic mean of precision and recall.
* **Confusion Matrix**: Detailed breakdown of correct and incorrect predictions.
* **ROC Curve**: Visual representation of the model's ability to distinguish between classes.
These metrics provide insights into the model's ability to correctly classify malignant and benign breast cancer samples. Additionally, visualizations such as confusion matrices were created to compare the predicted labels against the actual labels.

**Findings and Results Interpretation**
------------------------------------

## CNN Model

CNN emerged as the most effective model in this study, demonstrating superior performance in classifying breast cancer images. This can be attributed to its deep architecture, which allows it to extract complex features from the images, leading to more accurate predictions.

![image](https://github.com/user-attachments/assets/138ce00e-9aad-472b-9d0e-984de76bb300)


confusion matrix shows that the model has a strong ability to correctly identify malignant cases, as evidenced by the low number of false negatives (14). This is a crucial strength for a breast cancer classification model, as early detection is critical for successful treatment


![image](https://github.com/user-attachments/assets/9db3c7fd-1012-40dc-ba1a-cea3fb9cdc4d)

The closer the curve follows the left-hand border and then the top border of the graph towards the top-left corner, the more accurate the test.
 A larger AUC value indicates a better model. In this case, the AUC for both classes is 0.96, suggesting excellent discriminative power.
The ROC curve illustrates the trade-off between sensitivity and specificity. By adjusting the classification threshold, this can prioritize sensitivity (detecting more positive cases) or specificity (reducing false positives).

The model demonstrates strong performance in distinguishing between the two classes, as evidenced by the high AUC values.
The ROC curves for both classes are close to the top-left corner, indicating a good balance between sensitivity and specificity.


# **CONCLUSION**
----------

This project demonstrates the potential of machine learning models in improving breast cancer diagnosis in Kenya. By accurately classifying histopathological images into benign and malignant categories, the model can significantly reduce diagnostic errors and expedite the diagnostic process. This advancement is crucial for addressing the high breast cancer mortality rates in Kenya, where late-stage detection is prevalent.

The CNN model outperformed the other models in classifying breast cancer images, demonstrating higher accuracy, precision, and recall. The model's ability to effectively differentiate between benign and malignant cases holds promise for improving breast cancer diagnosis.

**Limitations**
------------

* The performance of the models is dependent on the quality and quantity of the training data.
* The current study focused on binary classification (benign vs. malignant); further research is needed to explore multi-class classification for different breast cancer subtypes.
* The generalization of the models to unseen data from different sources requires further evaluation.

**Key Challenges**

* Limited access to quality data and computational resources.
* Addressing ethical considerations related to medical data.
* Ensuring model interpretability and explainability.

# **RECOMMENDATIONS**
-----------------
1. **Integration into Healthcare Systems**: Integrate the developed model into the Kenyan healthcare system to assist pathologists and enhance diagnostic accuracy.
2. **Training and Education**: Provide training for healthcare professionals on the use and interpretation of the machine learning model to ensure effective implementation.
3. **Data Expansion**: Continuously expand and update the dataset with new histopathological images to improve the model's robustness and accuracy over time.
4. **Ethical Considerations**: Address ethical considerations related to patient data privacy and ensure compliance with relevant regulations.
5. **Public Awareness**: Increase public awareness about the importance of early breast cancer detection and the role of advanced diagnostic technologies in improving outcomes.
6. Tracking of key metrics, detecting model drift, and implementing updates or retraining as needed to adapt to changes in data or user behavior.



**Next Steps**
-------------

* Expand the dataset to include a larger and more diverse set of images.
* Develop a user-friendly interface for healthcare professionals to interact with the model.
* Conduct a pilot study in a clinical setting to assess the model's impact on patient outcomes.
* Model refinement
  


# **GROUP MEMBERS**

[Andrew Mutuku](https://github.com/AndrewNthale)\
[Amina Saidi]\
[Wambui Githinji]\
[Winnie Osolo]\
[Joseph Karumba](https://github.com/josephkarumba)\
[Margaret Njenga]