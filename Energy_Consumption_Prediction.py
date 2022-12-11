# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 10:31:37 2022

@author: Heather
"""


import pickle
import streamlit as st


#loading saved model
pickle_in = open('EnsembleModel.pkl', 'rb')
loaded_model = pickle.load(pickle_in)

# defining the function which will make the prediction using the data which the user inputs 
def prediction(building_id, meter, site_id, primary_use, floor_count, air_temperature, cloud_coverage, dew_temperature, hour, weekday, month):  
    
    # Making predictions 
   prediction = loaded_model.predict( 
       [[building_id, meter, site_id, primary_use, floor_count, air_temperature, cloud_coverage, dew_temperature, hour, weekday, month]])
       
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
    building_id = st.number_input("building_id") 
    meter = st.number_input("meter") 
    site_id = st.number_input("site_id")
    primary_use = st.number_input("primary_use")
    floor_count = st.number_input("floor_count")
    air_temperature = st.number_input("air_temperature")
    cloud_coverage = st.number_input("cloud_coverage")
    dew_temperature = st.number_input("dew_temperature")
    hour = st.number_input("hour")
    weekday = st.number_input("weekday")
    month = st.number_input("month")
    
    #code for prediction
    result =""
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = prediction(building_id, meter, site_id, primary_use, floor_count, air_temperature, cloud_coverage, dew_temperature, hour, weekday, month) 
    st.success('Your future Energy consumption for the building is {}'.format(result))
        #print(result)
     
if __name__=='__main__': 
    main()
    