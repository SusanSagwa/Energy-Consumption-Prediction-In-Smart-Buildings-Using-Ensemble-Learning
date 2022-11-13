# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 10:31:37 2022

@author: Heather
"""


import pickle
import streamlit as st


#loading saved model
pickle_in = open('C:/Users/Heather/OneDrive/Documents/EcModel/EnsembleModel.pkl', 'rb')
loaded_model = pickle.load(pickle_in)

# defining the function which will make the prediction using the data which the user inputs 
def prediction(lights, t1, rh1, t2, rh2, t3, rh3, t4, rh4, t5, rh5, rh6, t7, rh7, t8, rh8, rh9, tout, press_mm_hg, rhout, windspeed,tdewpoint):  
    
    # Making predictions 
   prediction = loaded_model.predict( 
       [[lights, t1, rh1, t2, rh2, t3, rh3, t4, rh4, t5, rh5, rh6, t7, rh7, t8, rh8, rh9, tout, press_mm_hg, rhout, windspeed,tdewpoint]])
       
   if prediction == 0:
       pred = 'Rejected'
   else:
       pred = 'Approved'
   return pred
 

# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:maroon;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Energy Consumption Prediction System</h1> 
    </div> 
    """
     
    #st.title("Energy Consumption Prediction System")
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # following lines create boxes in which user can enter data required to make prediction 
    lights = st.number_input("Lights") 
    t1 = st.number_input("Temperature of Kitchen") 
    rh1 = st.number_input("Humidity of Kitchen")
    t2 = st.number_input("Temperature of Living Room")
    rh2 = st.number_input("Humidity of Living Room")
    t3 = st.number_input("Temperature of Laundry room area")
    rh3 = st.number_input("Humidity of Laundry room area")
    t4 = st.number_input("Temperature of office room")
    rh4 = st.number_input("Humidity of office room")
    t5 = st.number_input("Temperature of bathroom")
    rh5 = st.number_input("Humidity of bathroom")
    rh6 = st.number_input("Humidity of Outside the building")
    t7 = st.number_input("Temperature of Ironing room")
    rh7 = st.number_input("Humidity of Ironing room")
    t8 = st.number_input("Temperature of teenager room")
    rh8 = st.number_input("Humidity of teenager room")
    rh9 = st.number_input("Humidity of Parent room")
    tout = st.number_input("Temperature of outside")
    press_mm_hg = st.number_input("Pressure")
    rhout = st.number_input("Humidity of outside")
    windspeed = st.number_input("Windspeed")
    tdewpoint = st.number_input("dew point")
    result =""
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = prediction(lights, t1, rh1, t2, rh2, t3, rh3, t4, rh4, t5, rh5, rh6, t7, rh7, t8, rh8, rh9, tout, press_mm_hg, rhout, windspeed,tdewpoint) 
        st.success('Your future Energy consumption for the building is {}'.format(result))
        print(prediction)
     
if __name__=='__main__': 
    main()
    