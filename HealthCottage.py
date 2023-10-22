import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
from tensorflow.keras.utils import img_to_array, load_img
import io
import keras
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import load_model

# Load the trained model
model_path = "C:/Users/hiler/Desktop/Professional Internship/HealthCottage/heart_disease_model_check.pkl"
with open(model_path, 'rb') as model_file:
    rf_model = pickle.load(model_file)

def preprocess_input(input_dict):
        # Map "Female" and "Male" to 1 and 0
        for key, value in input_dict.items():
            if isinstance(value, str):
                input_dict[key] = 1 if value.lower() == "Female" else 0
        return input_dict
    
heart=pd.read_csv("C:/Users/hiler/Desktop/Professional Internship/HealthCottage/heart.csv")
def predict_heart_disease(input_dict):

    # Convert input data to a DataFrame
    input_df = pd.DataFrame([input_dict])
    input_df=preprocess_input(input_df)

    # Make a prediction using the loaded model
    prediction=rf_model.predict(input_df)
    prediction_prob = rf_model.predict_proba(input_df)

    return prediction,prediction_prob[0]

def home_page():
        st.subheader("HealthCottage: Homepage")
        st.write("Welcome to the Heart and Malaria Disease Prediction App!")
        st.write("I'm Hileri Shah, and I present HealthCottage, your personal health and wellness companion. This cozy app offers disease prediction, heart health monitoring, and more. It's designed for simplicity and accuracy. With HealthCottage, I aim to empower you to take control of your well-being. HealthCottage is your path to healthier living, and I'm excited to share it with you.")
        st.write("Welcome to a cozy, tech-driven health experience at HealthCottage")
        st.write("Select an option from the sidebar to make a prediction.")
        img = Image.open("C:/Users/hiler/Desktop/Professional Internship/HealthCottage/HealthCottage4.png")
        st.image(img, width=500)

def main():
    st.header("HealthCottage: Your Home for Personalized Health and Wellness")
   # st.image("C:/Users/hiler/Desktop/Professional Internship/HealthCottage/heartimg.jpeg", width=500, use_column_width=False)
   # st.subheader("Home Page")
   # st.write("Welcome to the Heart Disease Prediction App!")
   # st.write("I'm Hileri Shah, and I present HealthCottage, your personal health and wellness companion. This app estimates your chance of heart disease (yes/no) using machine learning.")
   # st.write("Select an option from the sidebar to make a prediction.")
    img = Image.open("C:/Users/hiler/Desktop/Professional Internship/HealthCottage/HealthCottage4.png") 
    st.sidebar.image(img, width=300)
    st.sidebar.header("Options")
    selected_page = st.sidebar.radio("Choose a prediction type:", ["Home Page", "Heart Disease Prediction", "Malaria Disease Prediction", "Information Page"])
    img = Image.open("C:/Users/hiler/Desktop/Professional Internship/HealthCottage/sidebarimg.jpeg")
    st.sidebar.image(img, width=200)

    if selected_page == "Home Page":
            home_page()
    elif selected_page == "Heart Disease Prediction":
            heart_disease_page()
    elif selected_page == "Malaria Disease Prediction":
            malaria_disease_page()
    elif selected_page == "Information Page":
            information_page()

def heart_disease_page():
    st.header("Heart Disease Prediction")
    img = Image.open("C:/Users/hiler/Desktop/Professional Internship/HealthCottage/heartimg.jpeg")
    st.image(img, width=200)
    st.markdown("""
    Did you know that machine learning models can help you
    predict heart disease accurately? In this app, you can
    estimate your chance of heart disease (yes/no) in seconds!
    
    Here, a KNN model using an undersampling technique was constructed
    using survey data. To predict your heart disease status, follow the steps below:
    1. Enter your information;
    2. Press the "Predict" button.""")
    
    st.info("0 - No Heart Disease")
    st.info("1 - Possible Heart Disease. Consult a doctor!")

    def user_input():
        Age = st.slider("Age", min_value=0, max_value=100, step=1)
        Sex = st.radio("Sex", ["Female", "Male"])
        CP = st.selectbox("Chest Pain Type", ["Normal", "Having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)", "Showing probable or definite left ventricular hypertrophy by Estes' criteria"])
        TrestBPS = st.slider("Resting Blood Pressure (mm Hg)", min_value=0, max_value=200, step=1)
        Chol = st.slider("Cholesterol (mg/dl)", min_value=0, max_value=600, step=1)
        FBS = st.radio("Fasting Blood Sugar (> 120 mg/dl)", ["Yes", "No"])
        RestECG = st.selectbox("Resting ECG Results",  ["Typical Angina", "Atypical Angina", "Non-anginal pain", "Asymptomatic"])
        MaxHR = st.slider("Maximum Heart Rate Achieved", min_value=0, max_value=300, step=1)
        ExAng = st.radio("Exercise-Induced Angina", ["Yes", "No"])
        OldPeak = st.slider("Old Peak (ST depression induced by exercise)", min_value=0.0, max_value=6.2, step=0.1)
        Slope = st.selectbox("Slope of the Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
        CA = st.selectbox("Number of Major Vessels (0-3) Colored by Flourosopy", heart["ca"].unique())
        Thal = st.selectbox("Thalassemia Type", ["Normal", "Fixed defect", "Reversable defect"])

        # Define mappings for string columns
        sex_mapping = {"Female": 0, "Male": 1}
        ecg_mapping = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal pain": 2, "Asymptomatic": 3}
        cp_mapping = {"Normal" : 0, "Having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)" : 1, "Showing probable or definite left ventricular hypertrophy by Estes' criteria" : 2}
        slope_mapping = {"Upsloping" : 0, "Flat" : 1, "Downsloping" : 2}
        fbs_mapping = {"Yes": 1, "No": 0}
        exang_mapping = {"Yes": 1, "No": 0}
        thal_mapping = {"Normal": 0, "Fixed defect":1, "Reversable defect" : 2}

        # Convert string values to integers
        Sex = sex_mapping[Sex]
        CP = cp_mapping[CP]
        RestECG = ecg_mapping[RestECG]
        FBS = fbs_mapping[FBS]
        ExAng = exang_mapping[ExAng]
        Slope = slope_mapping[Slope]
        Thal = thal_mapping[Thal]

        input_data = {
            "age": Age,
            "sex": Sex,
            "cp": CP,
            "trestbps": TrestBPS,
            "chol": Chol,
            "fbs": FBS,
            "restecg": RestECG,
            "thalach": MaxHR,
            "exang": ExAng,
            "oldpeak": OldPeak,
            "slope": Slope,
            "ca": CA,
            "thal": Thal
        }

        return input_data

    input_data = user_input()

    if st.button("Predict"):
        prediction,prediction_prob = predict_heart_disease(input_data)
        if prediction == [0]:
            st.markdown(f"**Prediction of Heart Disease: {'No Heart Disease'}**")
            st.markdown(f"**Probability of Heart Disease: {prediction_prob[1] * 100:.2f}%**")
        else:
            st.markdown(f"**Prediction of Heart Disease: {'You may have Heart Disease. Go consult a doctor'}**")
            st.markdown(f"**Probability of Heart Disease: {prediction_prob[1] * 100:.2f}%**")

def malaria_disease_page():
    st.header("Malaria Disease Prediction")
    img = Image.open("C:/Users/hiler/Desktop/Professional Internship/HealthCottage/malariaimg.jpeg")
    st.image(img, width=200)

    def load_image(image_file):
        img = Image.open(image_file)
        return img

    st.write("Upload your image for identification")
    image_inp = st.file_uploader("Upload your image for identification", type=["png", "jpg", "jpeg"])

    if image_inp is not None:
        img_input = st.image(load_image(image_inp), width=250)

        # Load the trained model
        model = tf.keras.models.load_model('C:/Users/hiler/Desktop/Professional Internship/HealthCottage/Malaria_retrain/malaria_model.h5')

        # Load and preprocess the new image you want to predict
        new_image_path = image_inp
        img = load_img(new_image_path, target_size=(128, 128))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0  # Normalize the image

        # Make the prediction
        prediction = model.predict(img_array)

        # Interpret the prediction
        if prediction[0][0] > 0.5:
            st.success("The image contains an uninfected cell (Malaria negative).")
        else:
            st.error("The image contains a parasitized cell (Malaria positive).")
    else:
        st.info("Please upload an image for identification")

def information_page():
        st.title('Information Page')
        img = Image.open("C:/Users/hiler/Desktop/Professional Internship/HealthCottage/infoimg.jpeg")
        st.image(img, width=200)
        st.title('Information on Malaria')
        st.header('Diagnosis')
        st.write('To diagnose malaria, your doctor will likely review your medical history and recent travel, conduct a physical exam, and order blood tests.')
        st.write('')
        st.write('Blood tests can indicate:')
        st.write('')
        st.write('1. The presence of the parasite in the blood, to confirm that you have malaria')
        st.write('')
        st.write('2. Which type of malaria parasite is causing your symptoms')
        st.write('')
        st.write('3. If your infection is caused by a parasite resistant to certain drugs')
        st.write(' ')
        st.write('4. Whether the disease is causing any serious complications')
        st.write(' ') 
        st.header('Treatment')
        st.write('Malaria is treated with prescription drugs to kill the parasite. The types of drugs and the length of treatment will vary, depending on:')
        st.write(' ')
        st.write('1. Which type of malaria parasite you have')
        st.write('')
        st.write('2. The severity of your symptoms')
        st.write('')
        st.write('3. Your age')
        st.write(' ')
        st.write('4. Whether you are pregnant')
        st.write(' ') 
        st.subheader('Medications')
        st.write('The most common antimalarial drugs include:')
        st.write(' ') 
        st.write('Chloroquine phosphate:')
        st.write('Chloroquine is the preferred treatment for any parasite that is sensitive to the drug. But in many parts of the world, parasites are resistant to chloroquine, and the drug is no longer an effective treatment.')
        st.write(' ') 
        st.write('Artemisinin-based combination therapies (ACTs):')
        st.write('Artemisinin-based combination therapy (ACT) is a combination of two or more drugs that work against the malaria parasite in different ways. This is usually the preferred treatment for chloroquine-resistant malaria. Examples include artemether-lumefantrine (Coartem) and artesunate-mefloquine.')
        st.write(' ')
        st.write('Other common antimalarial drugs include:')
        st.write('- Atovaquone-proguanil (Malarone)')                    
        st.write('- Quinine sulfate (Qualaquin) with doxycycline (Oracea, Vibramycin, others)')
        st.write('- Primaquine phosphate')
        st.write(' ')
        st.warning('Before taking any of these medicines, be sure to consult your doctor first to know if you are allergic to anything in the medicine or which drug suits you the best!')
        st.info('Afterall, your health is our top priority! :)')
        st.title('')
        st.title('Information on Heart Diseases')
        st.header('Diagnosis')
        st.write('Your health care provider will examine you and ask about your personal and family medical history. Your health care provider is likely to ask you questions, such as:')
        st.write('When did your symptoms begin?')
        st.write('Do you always have symptoms or do they come and go?')
        st.write('How severe are your symptoms?')
        st.write('What, if anything, seems to improve your symptoms?')
        st.write('What, if anything, makes your symptoms worse?')
        st.write('Do you have a family history of heart disease, diabetes, high blood pressure or other serious illness?')
        st.write('')
        st.write('Many different tests are used to diagnose heart disease. Besides blood tests and a chest X-ray, tests to diagnose heart disease can include:')
        st.write(' ') 
        st.write('Electrocardiogram (ECG or EKG):')
        st.write('An ECG is a quick and painless test that records the electrical signals in the heart. It can tell if the heart is beating too fast or too slowly.')
        st.write(' ') 
        st.write('Holter Monitoring:')
        st.write('A Holter monitor is a portable ECG device that is worn for a day or more to record the activity of the heart during daily activities. This test can detect irregular heartbeats that are not found during a regular ECG exam.')
        st.write(' ')
        st.write('Echocardiogram:')
        st.write('This noninvasive exam uses sound waves to create detailed images of the heart in motion. It shows how blood moves through the heart and heart valves. An echocardiogram can help determine if a valve is narrowed or leaking.')                    
        st.write('') 
        st.write('Exercise tests or stress tests:')
        st.write('These tests often involve walking on a treadmill or riding a stationary bike while the heart is monitored. Exercise tests help reveal how the heart responds to physical activity and whether heart disease symptoms occur during exercise. If you cannot exercise, you might be given medications.')
        st.write(' ') 
        st.write('Cardiac catheterization:')
        st.write('This test can show blockages in the heart arteries. A long, thin flexible tube (catheter) is inserted in a blood vessel, usually in the groin or wrist, and guided to the heart. Dye flows through the catheter to arteries in the heart. The dye helps the arteries show up more clearly on X-ray images taken during the test.')
        st.write(' ')
        st.write('Heart (cardiac) CT scan:')
        st.write('In a cardiac CT scan, you lie on a table inside a doughnut-shaped machine. An X-ray tube inside the machine rotates around your body and collects images of your heart and chest.')                    
        st.write('')
        st.write('Heart (cardiac) magnetic resonance imaging (MRI) scan:')
        st.write('A cardiac MRI uses a magnetic field and computer-generated radio waves to create detailed images of the heart.')                    
        st.write('')
        st.header('Treatment')
        st.write('Heart disease treatment depends on the cause and type of heart damage. Healthy lifestyle habits — such as eating a low-fat, low-salt diet, getting regular exercise and good sleep, and not smoking — are an important part of treatment.')
        st.subheader('Medications')
        st.write('If lifestyle changes alone do not work, medications may be needed to control heart disease symptoms and to prevent complications. The type of medication used depends on the type of heart disease.')
        st.subheader('Surgery or other procedures')
        st.write('Some people with heart disease may need a procedure or surgery. The type of procedure or surgery will depend on the type of heart disease and the amount of damage to the heart.')
        st.header('Lifestyle and home remedies')
        st.write('Heart disease can be improved — or even prevented — by making certain lifestyle changes.')
        st.write('')
        st.write('The following changes are recommended to improve heart health:')
        st.write('')
        st.write('Do not smoke:')
        st.write('Smoking is a major risk factor for heart disease, especially atherosclerosis. Quitting is the best way to reduce the risk of heart disease and its complications. If you need help quitting, talk to your provider.')
        st.write(' ') 
        st.write('Eat healthy food:')
        st.write('Eat plenty of fruits, vegetables and whole grains. Limit sugar, salt and saturated fats.')
        st.write(' ')
        st.write('Control blood pressure:')
        st.write('Uncontrolled high blood pressure increases the risk of serious health problems. Get your blood pressure checked at least every two years if you are 18 and older. If you have risk factors for heart disease or are over age 40, you may need more-frequent checks. Ask your health care provider what blood pressure reading is best for you.')                    
        st.write('') 
        st.write('Get a cholestrol test:')
        st.write('Ask your provider for a baseline cholesterol test when you are in your 20s and then at least every 4 to 6 years. You may need to start testing earlier if high cholesterol is in your family. You may need more-frequent checks if your test results are not in a desirable range or you have risk factors for heart disease.')
        st.write(' ') 
        st.write('Manage diabetes:')
        st.write('If you have diabetes, tight blood sugar control can help reduce the risk of heart disease.')
        st.write(' ')
        st.write('Exercise:')
        st.write('Physical activity helps you achieve and maintain a healthy weight. Regular exercise helps control diabetes, high cholesterol and high blood pressure — all risk factors for heart disease. Aim for 30 to 60 minutes of physical activity most days of the week. Talk to your health care provider about the amount and type of exercise that is best for you.')                    
        st.write('')
        st.write('Maintain a healthy weight:')
        st.write('Being overweight increases the risk of heart disease. Talk with your care provider to set realistic goals for body mass index (BMI) and weight.')                    
        st.write('')
        st.write('Manage stress:')
        st.write('Find ways to help reduce emotional stress. Getting more exercise, practicing mindfulness and connecting with others in support groups are some ways to reduce and manage stress. If you have anxiety or depression, talk to your provider about strategies to help.')
        st.write(' ')
        st.write('Practice good hygiene:')
        st.write('Regularly wash your hands and brush and floss your teeth to keep yourself healthy.')                    
        st.write('')
        st.write('Practice good sleep habits:')
        st.write('Poor sleep may increase the risk of heart disease and other chronic conditions. Adults should aim to get 7 to 9 hours of sleep daily. Kids often need more. Go to bed and wake at the same time every day, including on weekends. If you have trouble sleeping, talk to your provider about strategies that might help.')                    
        st.write('')
        st.info('It is never too early to make healthy lifestyle changes, such as quitting smoking, eating healthy foods and becoming more physically active. A healthy lifestyle is the main protection against heart disease and its complications.')
            

if __name__ == "__main__":
    main()
