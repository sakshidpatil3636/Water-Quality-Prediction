# **Water Quality Prediction**

## **Project Description**  
The **Water Quality Prediction** website enables users to input various parameters related to water quality, such as:  
- pH levels  
- Hardness  
- Solids  
- Chloramines  
- Sulfate  
- Conductivity  
- Organic Carbon  
- Trihalomethanes  
- Turbidity  

Based on these inputs, a machine learning model (Random Forest Classifier) determines whether the water is safe for drinking.  

This project leverages machine learning for environmental and health safety, providing users with an easy-to-use interface to analyze water quality.  

---

## **Technologies and Tools Used**  
- **Programming Language**: Python  
- **Machine Learning Model**: Random Forest Classifier  
- **Frontend**: HTML  
- **Backend**: Flask (for deployment)  
- **Model Serialization**: Pickle (for saving and loading the trained model)  

---

## **Features**  
- **User Input Interface**: Users can input values for multiple water quality parameters.  
- **Prediction Capability**: The trained Random Forest Classifier predicts whether the water is drinkable or not.  
- **Model Deployment**: The model is deployed using Flask for real-time predictions.  
- **Reusable Model**: The trained model is serialized and deserialized using Pickle.  

---

## **How It Works**  
1. **User Inputs**: Users enter values for water parameters such as pH, hardness, solids, etc.  
2. **Prediction**: The model processes the input and predicts whether the water is safe for drinking.  
3. **Results Displayed**: The result (Good or Not Good for Drinking) is shown on the website.  

---

## **Technological Workflow**  
1. **Model Creation**:  
   - The Random Forest Classifier is developed using Python.  
   - The model is trained and tuned using a labeled dataset to optimize prediction accuracy.  

2. **Model Serialization**:  
   - The trained model is serialized using Pickle for future use.  

3. **Web Application**:  
   - The website frontend is created using HTML.  
   - Flask is used to build and deploy the backend application.  

4. **Deployment**:  
   - The application is hosted locally, allowing real-time user interaction and prediction.  

---

## **Future Enhancements**  
- Add visualization of water quality parameters for better insights.  
- Deploy the application on a cloud platform for global accessibility.  
- Incorporate more advanced machine learning models to improve accuracy.  

---

## **Contributors**  
- **Sakshi Dinkar Patil**  

---

